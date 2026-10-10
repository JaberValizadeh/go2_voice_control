#!/usr/bin/env python3
"""Translate recognized speech into Go2 Twist commands.

Turns are closed-loop on the Gazebo yaw rather than timed, because the gait
delivers only part of the commanded angular rate (about 0.34 rad/s for a 0.6
command). A timed turn would fall well short; watching the actual heading makes
the turn land regardless of how well the gait tracks.

Stairs are the other thing this node does, and they are not a walk with a
different speed. A stair run puts the gait into its climbing mode - a crawl,
feet that remember their own ground, a trunk levelled against its own pitch -
reads the vertical scanner on the muzzle to measure each riser, and lifts the
feet by exactly what that riser turned out to be. The pieces are:

    perception.py   scan line -> how far ahead the next riser is, and how tall
    stepper.py      one riser at a time: look, creep in, step over, settle
    stairs.py       the errand around it: approach, square up, traverse, stop

and this node is the wiring between them and ROS.
"""

import math

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Point32, Twist
from sensor_msgs.msg import LaserScan, PointCloud
from std_msgs.msg import Bool, Float64, String

from go2_voice_control import interpreter, perception, stairs, stepper


class VoiceTeleop(object):

    def __init__(self):
        self.model_name = rospy.get_param("~model_name", "unitree_go2")
        self.max_linear = float(rospy.get_param("~max_linear_speed", 0.4))
        self.max_angular = float(rospy.get_param("~max_angular_speed", 0.8))
        self.min_linear = float(rospy.get_param("~min_linear_speed", 0.05))
        self.min_angular = float(rospy.get_param("~min_angular_speed", 0.15))

        # Clamp the configured speeds into range rather than trusting them: a
        # linear_speed above max_linear_speed would otherwise be honoured until
        # the first "faster", which then silently lowers it.
        self.linear_speed = min(self.max_linear, max(
            self.min_linear, float(rospy.get_param("~linear_speed", 0.2))))
        # Reverse walking has a different mechanical sweet spot from forward
        # walking on this model.  Keep it independently tunable so raising the
        # forward speed does not disturb the already-stable reverse gait.
        self.backward_speed = min(self.max_linear, max(
            self.min_linear, float(rospy.get_param(
                "~backward_speed", self.linear_speed))))
        self.angular_speed = min(self.max_angular, max(
            self.min_angular, float(rospy.get_param("~angular_speed", 0.4))))

        self.timeout = float(rospy.get_param("~command_timeout", 5.0))
        self.publish_rate = float(rospy.get_param("~publish_rate", 10.0))
        self.turn_tolerance = math.radians(
            float(rospy.get_param("~turn_tolerance_deg", 3.0)))

        # Where the flight is. Given rather than discovered: a horizontal
        # lidar at knee height sees a rising surface as a wall, so this robot
        # cannot find a staircase by looking for one. In a known room, telling
        # it is honest.
        self.flight = stairs.Flight(
            foot=rospy.get_param("~stair_foot", [2.4, 0.0]),
            top=rospy.get_param("~stair_top", [4.6, 0.0]),
            bottom_z=float(rospy.get_param("~stair_bottom_z", 0.30)),
            top_z=float(rospy.get_param("~stair_top_z", 0.75)),
            heading=float(rospy.get_param("~stair_heading", 0.0)))
        self.stair_run = None

        # Seeing the steps. Disabling the scanner explicitly selects the
        # open-loop ramp traverse; a requested scanner may still be starting.
        self.use_scanner = bool(rospy.get_param("~use_stair_scanner", True))
        self.scan_max = float(rospy.get_param("~stair_scan_max", 4.0))
        self.look_ahead = float(rospy.get_param("~stair_look_ahead", 1.6))
        self.min_rise = float(rospy.get_param("~stair_min_rise", 0.025))
        self.clearance = float(rospy.get_param("~stair_clearance", 0.045))
        self.max_swing = float(rospy.get_param("~stair_max_swing", 0.18))
        self.profile = None
        self.scan_stamp = None
        self.pitch = 0.0
        self.roll = 0.0
        self.stair_mode = False
        self.height_offset = 0.0

        self.twist = Twist()
        self.yaw = None
        self.position = None
        self.height = None
        self.target_yaw = None
        self.turn_direction = 0.0
        self.turn_deadline = None
        self.forward_after_turn = False
        self.last_command = rospy.Time.now()

        self.publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self.tread_publisher = rospy.Publisher("stair_treads", PointCloud, queue_size=1)
        self.heading_publisher = rospy.Publisher("stair_heading", Float64,
                                                 queue_size=1, latch=True)
        self.heading_publisher.publish(Float64(self.flight.heading))
        # The gait's climbing mode, and the foot clearance to climb with.
        # Latched, so a gait node that comes up late still learns it is not in
        # stair mode rather than inheriting whatever the last run left.
        self.mode_publisher = rospy.Publisher("stair_mode", Bool,
                                              queue_size=1, latch=True)
        self.swing_publisher = rospy.Publisher("swing_height", Float64,
                                               queue_size=1)
        # What the robot thinks it can see, in words. Nothing depends on it;
        # it exists because "it is climbing" and "it is stuck against a riser
        # it never saw" look identical from outside.
        self.profile_publisher = rospy.Publisher("~stair_profile", String,
                                                 queue_size=1)
        rospy.Subscriber("voice_commands", String, self.on_command,
                         queue_size=10)
        rospy.Subscriber("gazebo/model_states", ModelStates,
                         self.on_model_states, queue_size=1)
        rospy.Subscriber("stair_scan", LaserScan, self.on_scan, queue_size=1)
        rospy.Subscriber("body_height_offset", Float64,
                         self.on_height_offset, queue_size=1)
        self.mode_publisher.publish(Bool(False))

    # ------------------------------------------------------------------ state

    def on_model_states(self, message):
        try:
            index = message.name.index(self.model_name)
        except ValueError:
            return
        pose = message.pose[index]
        q = pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # Pitch is not decoration here: the scanner reads a riser through it,
        # and the robot is nose-up by a riser over its wheelbase for the whole
        # of a climb. See perception.scan_points.
        self.pitch = math.asin(
            max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x))))
        self.roll = math.atan2(2 * (q.w * q.x + q.y * q.z),
                               1 - 2 * (q.x * q.x + q.y * q.y))
        self.position = (pose.position.x, pose.position.y)
        # Height is what a stair run is actually measured against - see
        # go2_voice_control/stairs.py.
        self.height = pose.position.z

    def on_scan(self, message):
        """Turn one sweep of the muzzle scanner into a staircase."""
        self.profile = perception.from_scan(
            message.angle_min, message.angle_increment, message.ranges,
            pitch=self.pitch,
            range_min=max(0.02, message.range_min),
            range_max=min(self.scan_max, message.range_max),
            look_ahead=self.look_ahead, min_rise=self.min_rise)
        self.scan_stamp = rospy.Time.now()
        if self.position is not None and self.height is not None and abs(self.roll) < 0.10:
            cloud = PointCloud()
            cloud.header.stamp = self.scan_stamp
            cloud.header.frame_id = "world"
            for start, end, z in self.profile.footholds():
                for x in (start, end):
                    cloud.points.append(Point32(
                        self.position[0] + x * math.cos(self.yaw),
                        self.position[1] + x * math.sin(self.yaw), self.height + z))
            self.tread_publisher.publish(cloud)

    def on_height_offset(self, message):
        """Keep the terrain-height datum fixed while the gait crouches.

        The gait reports its current change from the normal stance, including
        the transition ramp. Adding it to trunk height lets the height tests
        measure climbing rather than a deliberate change in leg extension.
        """
        if math.isfinite(message.data):
            self.height_offset = message.data

    @property
    def scanner_alive(self):
        """A scan seen recently enough to steer a climb by."""
        if self.scan_stamp is None:
            return False
        return (rospy.Time.now() - self.scan_stamp).to_sec() < 1.0

    @property
    def turning(self):
        return self.target_yaw is not None or self.turn_deadline is not None

    @property
    def on_the_stairs(self):
        return self.stair_run is not None

    # ----------------------------------------------------------------- stairs

    def start_stairs(self, direction):
        if self.position is None or self.yaw is None:
            rospy.logwarn("No pose yet - cannot start a stair run.")
            return
        self.cancel_turn()
        self.cancel_stairs()
        self.twist = Twist()
        sequencer = None
        if self.use_scanner:
            if not self.scanner_alive:
                rospy.logwarn("No stair scanner on /stair_scan yet - "
                              "the traverse will wait for fresh scans.")
            sequencer = stepper.StepSequencer(
                direction=direction,
                clearance=self.clearance,
                max_swing=self.max_swing,
                min_rise=self.min_rise,
                # Several supported foot placements are needed before a
                # full riser appears in trunk height during a slow crawl.
                lift_timeout=float(rospy.get_param("~stair_lift_timeout", 30.0)))
        walkup = float(rospy.get_param("~stair_walkup_speed", 0.0))
        self.stair_run = stairs.StairRun(
            self.flight, direction,
            max_linear=self.max_linear,
            max_angular=self.max_angular,
            traverse_speed=float(rospy.get_param("~stair_speed", 0.30)),
            # 0 or less keeps the walk-up at the crawl creep; see stairs.py.
            walkup_speed=(walkup if walkup > 0.0 else None),
            descent_speed=float(rospy.get_param(
                "~stair_descent_speed", 0.18)),
            sequencer=sequencer,
            # A stepped climb corrects sideways rather than by turning; see
            # StairRun. Only when there is a sequencer, because that is the
            # case where the robot is on a narrow flight rather than a ramp.
            strafe_gain=(float(rospy.get_param("~stair_strafe_gain", 0.6))
                         if sequencer else 0.0),
            max_strafe=float(rospy.get_param("~stair_max_strafe", 0.12)),
            timeout=float(rospy.get_param(
                "~stair_timeout", 360.0 if sequencer else 120.0)),
            # A stepped climb spends most of its time deliberately stationary
            # - measuring, and letting the rear feet catch up - so the stall
            # detector has to be given room to tell that apart from a robot
            # scuffing a nosing. It still ends on height, not on the clock.
            stall_seconds=float(rospy.get_param(
                "~stair_stall_seconds", 25.0 if sequencer else 8.0)))
        rospy.loginfo("Stair run: going %s, %s",
                      "up" if direction > 0 else "down",
                      "step by step on the scanner" if sequencer
                      else "open loop")

    def cancel_stairs(self):
        if self.stair_run is not None:
            self.set_stair_mode(False)
        self.stair_run = None

    def set_stair_mode(self, wanted):
        """Latch the crawl after a measured riser until the run ends.

        The approach uses the normal walk. Once climbing starts, temporary
        loss of a visible riser must not release the stair gait: the scanner
        looks ahead of the feet and can see only flat platform at the top.
        The gait finishes an active landing before leaving crawl mode and
        holds its supporting footholds if stopped on uneven treads.
        """
        if wanted == self.stair_mode:
            return
        self.stair_mode = wanted
        self.mode_publisher.publish(Bool(wanted))

    def update_stairs(self):
        """Drive one step of a stair run, if one is in progress."""
        if self.stair_run is None:
            return
        if self.position is None or self.yaw is None or self.height is None:
            return
        pose = (self.position[0], self.position[1], self.yaw,
                self.height + self.height_offset)
        profile = self.profile if self.scanner_alive else None
        linear, angular = self.stair_run.update(
            pose, rospy.Time.now().to_sec(), profile)
        # Apply climbing clearance on the tick that commits to the riser.
        # LIFT can begin just before the trunk enters the flight margin,
        # because the front feet and scanner reach ahead of the trunk.
        sequencer = self.stair_run.sequencer
        if not self.stair_mode and sequencer is not None \
                and self.stair_run.state == stairs.TRAVERSE \
                and sequencer.riser \
                and (self.flight.on_flight(self.position)
                     or sequencer.state == stepper.LIFT):
            # The wide 0.35 m margin is deliberate, and tightening it to save
            # the placements spent crawling the flat floor in front of the
            # flight does not pay: at 0.06 m the robot trots most of the way
            # to the riser and changes gait almost on top of it, and a climb
            # that had gone three for three lost its balance on the third
            # step. The crawl wants to be established and settled before the
            # first riser, not arriving with it.
            self.set_stair_mode(True)
        self.twist = Twist()
        self.twist.linear.x = linear
        self.twist.linear.y = self.stair_run.lateral
        self.twist.angular.z = angular
        plan = self.stair_run.plan
        if plan is not None:
            # Republished every cycle rather than on change: the gait clamps
            # it to what the leg can actually reach, and a gait node restarted
            # mid-climb would otherwise keep the flat-ground swing height.
            self.swing_publisher.publish(Float64(plan.swing_height))
        self.report_profile()
        if self.stair_run.finished:
            rospy.loginfo("Stair run finished: %s (z = %.2f m)",
                          self.stair_run.describe(), self.height)
            self.twist = Twist()
            self.stair_run = None
            self.set_stair_mode(False)

    def report_profile(self):
        """Say what the robot can see and what it is doing about it."""
        if self.stair_run is None:
            return
        seen = self.profile.describe() if (
            self.profile is not None and self.scanner_alive) else "nothing"
        plan = self.stair_run.plan
        text = "%s | sees: %s" % (
            plan.note if plan is not None else self.stair_run.state, seen)
        self.profile_publisher.publish(String(text))
        rospy.loginfo_throttle(1.0, "Stairs: %s", text)

    # ------------------------------------------------------------------ turns

    def cancel_turn(self):
        self.target_yaw = None
        self.turn_deadline = None
        self.turn_direction = 0.0
        self.forward_after_turn = False

    def start_turn(self, direction, angle, forward_after=False):
        self.twist = Twist()
        self.turn_direction = direction
        self.forward_after_turn = forward_after
        if self.yaw is not None:
            self.target_yaw = interpreter.normalize_angle(
                self.yaw + direction * angle)
            self.turn_deadline = None
        else:
            # No pose feed. A timed turn under-rotates because the gait does not
            # reach the commanded rate, so treat this as a degraded fallback.
            rospy.logwarn("No Gazebo pose available; falling back to a timed "
                          "turn, which will under-rotate.")
            self.target_yaw = None
            self.turn_deadline = (rospy.Time.now() +
                                  rospy.Duration(angle / self.angular_speed))
        self.twist.angular.z = direction * self.angular_speed

    def finish_turn(self):
        drive_after = self.forward_after_turn
        self.cancel_turn()
        self.twist = Twist()
        if drive_after:
            self.twist.linear.x = self.linear_speed
            # Restart the timeout window. A half turn takes longer than
            # command_timeout, so without this the drive that the turn was
            # supposed to lead into is cancelled on the very next cycle.
            self.last_command = rospy.Time.now()
        rospy.loginfo("Turn complete%s",
                      "; moving forward" if drive_after else "")

    def update_turn(self):
        if self.target_yaw is not None and self.yaw is not None:
            remaining = interpreter.turn_remaining(self.target_yaw, self.yaw)
            if interpreter.turn_finished(remaining, self.turn_direction,
                                         self.turn_tolerance):
                self.finish_turn()
            else:
                self.twist.angular.z = interpreter.turn_speed(
                    remaining, self.turn_direction, self.angular_speed,
                    self.min_angular)
        elif (self.turn_deadline is not None
              and rospy.Time.now() >= self.turn_deadline):
            self.finish_turn()

    # --------------------------------------------------------------- commands

    def rescale_motion(self):
        """Re-apply the current speeds to whatever motion is in progress."""
        if self.twist.linear.x:
            speed = (self.linear_speed if self.twist.linear.x > 0.0
                     else self.backward_speed)
            self.twist.linear.x = math.copysign(speed, self.twist.linear.x)
        if self.twist.angular.z and not self.turning:
            self.twist.angular.z = math.copysign(self.angular_speed,
                                                 self.twist.angular.z)

    def apply(self, intent):
        """Act on an interpreted intent. Returns True if it was understood."""
        if intent.kind == interpreter.UNKNOWN:
            return False

        if intent.kind == interpreter.SPEED:
            self.linear_speed = interpreter.scale_speed(
                self.linear_speed, intent.scale, self.min_linear,
                self.max_linear)
            self.backward_speed = interpreter.scale_speed(
                self.backward_speed, intent.scale, self.min_linear,
                self.max_linear)
            self.angular_speed = interpreter.scale_speed(
                self.angular_speed, intent.scale, self.min_angular,
                self.max_angular)
            self.rescale_motion()
        elif intent.kind == interpreter.STOP:
            self.cancel_turn()
            self.cancel_stairs()
            self.twist = Twist()
        elif intent.kind == interpreter.STAIRS:
            self.start_stairs(intent.direction)
        elif intent.kind == interpreter.TURN:
            self.cancel_stairs()
            self.start_turn(intent.direction, intent.angle,
                            intent.forward_after)
        elif intent.kind == interpreter.WALK:
            self.cancel_turn()
            self.cancel_stairs()
            self.twist = Twist()
            speed = (self.linear_speed if intent.direction > 0.0
                     else self.backward_speed)
            self.twist.linear.x = intent.direction * speed
        return True

    def on_command(self, message):
        intent = interpreter.interpret(message.data)
        self.last_command = rospy.Time.now()
        if not self.apply(intent):
            rospy.logwarn("Unknown voice command: '%s'", message.data)
            return
        rospy.loginfo("Command '%s' -> %s | linear %.2f, angular %.2f",
                      intent.text, intent.kind, self.twist.linear.x,
                      self.twist.angular.z)

    # ------------------------------------------------------------------- loop

    def timed_out(self):
        # A stair run is a long errand issued by one word, so the idle timeout
        # must not cut it off half way up.
        if self.turning or self.on_the_stairs or self.timeout <= 0:
            return False
        return (rospy.Time.now() - self.last_command).to_sec() > self.timeout

    def spin_once(self):
        self.update_stairs()
        self.update_turn()
        if self.timed_out():
            self.twist = Twist()
        self.publisher.publish(self.twist)

    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            self.spin_once()
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        self.publisher.publish(Twist())


if __name__ == "__main__":
    rospy.init_node("go2_voice_teleop")
    try:
        VoiceTeleop().run()
    except rospy.ROSInterruptException:
        pass
