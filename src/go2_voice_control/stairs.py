#!/usr/bin/env python3
"""Getting the robot up a flight and back down again, when told to.

"Go up" is not a direction, it is an errand. The robot may be anywhere in the
room and pointing anywhere when it is asked, so the behaviour has four parts
and they have to happen in order:

    APPROACH   walk to a staging point in front of the flight
    ALIGN      turn to face along it
    TRAVERSE   walk forward until the height has actually changed
    ARRIVED    stop

Skipping the staging point and simply walking at the flight is what makes a
robot mount a staircase diagonally, catch one foot on a nosing and go over.
Approaching a fixed point first means the climb always starts square on.

The flight's geometry is given rather than discovered. A horizontal lidar at
knee height sees a rising surface as a wall - which is exactly what it looks
like from there - so a robot with this sensor suite cannot find a staircase by
looking for one. Telling it where the stairs are is honest for a voice demo in
a known room, and it is the same choice go2_person_search makes for its ramps.

The traverse ends on HEIGHT, not on distance. Distance would mean trusting that
the robot walked as far as it was told, and on a slope it does not - the feet
slip, the body pitches, and odometry along the floor is not the thing being
asked about anyway. "Am I 45 cm higher than I was" is the actual question.

Deliberately free of ROS, so the whole state machine can be exercised without
a simulator:

    python3 -m go2_voice_control.stairs
"""

import math

from go2_voice_control import stepper

APPROACH = "approach"
ALIGN = "align"
TRAVERSE = "traverse"
ARRIVED = "arrived"
FAILED = "failed"

UP = 1.0
DOWN = -1.0


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Flight(object):
    """One flight: where it starts, where it ends, and which way is up.

    `heading` is the bearing to walk to go UP. Going down is the same flight
    walked on the reciprocal, from the other end.
    """

    def __init__(self, foot, top, bottom_z, top_z, heading, stage=1.2):
        self.foot = tuple(foot)
        self.top = tuple(top)
        self.bottom_z = float(bottom_z)
        self.top_z = float(top_z)
        self.heading = float(heading)
        # How far back from the flight to stand before starting. Far enough to
        # be square on by the time the first foot lands on it.
        self.stage = float(stage)

    def staging_point(self, direction):
        """Where to stand before setting off, for this direction of travel."""
        if direction > 0:
            base, bearing = self.foot, self.heading + math.pi
        else:
            base, bearing = self.top, self.heading
        return (base[0] + self.stage * math.cos(bearing),
                base[1] + self.stage * math.sin(bearing))

    def travel_heading(self, direction):
        return self.heading if direction > 0 else normalize_angle(
            self.heading + math.pi)

    def target_height(self, direction):
        return self.top_z if direction > 0 else self.bottom_z

    def along(self, position):
        """How far along the flight a position is, from foot to top.

        Negative before the foot, greater than the run past the top. Used to
        tell "still walking to the stairs" from "on them", which the stall
        detector has to know: no height is expected before the first step.
        """
        dx = self.top[0] - self.foot[0]
        dy = self.top[1] - self.foot[1]
        run = math.hypot(dx, dy)
        if run < 1e-9:
            return 0.0
        return ((position[0] - self.foot[0]) * dx
                + (position[1] - self.foot[1]) * dy) / run

    def run_length(self):
        return math.hypot(self.top[0] - self.foot[0],
                          self.top[1] - self.foot[1])

    def cross_track(self, position):
        """Signed sideways offset from the flight's centre line, in metres.

        Positive to the left of the direction of travel up the flight. A
        traverse that only holds its heading has no idea it is doing this, and
        a flight is narrower than the room it is in.
        """
        return (-(position[0] - self.foot[0]) * math.sin(self.heading)
                + (position[1] - self.foot[1]) * math.cos(self.heading))

    def on_flight(self, position, margin=0.35):
        """Is the robot actually on the flight rather than approaching it?"""
        along = self.along(position)
        return -margin <= along <= self.run_length() + margin

    def height_reached(self, z, direction, tolerance=0.06):
        """Has the robot got where it was going, vertically?"""
        if direction > 0:
            return z >= self.top_z - tolerance
        return z <= self.bottom_z + tolerance

    def endpoint_reached(self, position, z, direction, margin=0.20,
                         tolerance=0.06):
        """Has the trunk cleared the flight and reached the target level?

        Height proves that the incline was climbed; progress beyond the end
        keeps the rear feet from being left on the slope when motion stops.
        """
        if not self.height_reached(z, direction, tolerance):
            return False
        along = self.along(position)
        if direction > 0:
            return along + 1e-9 >= self.run_length() + margin
        return along <= -margin + 1e-9


class StairRun(object):
    """The state machine, driven by pose and answering with a velocity."""

    def __init__(self, flight, direction, max_linear=0.45, max_angular=0.7,
                 arrive_radius=0.35, align_tolerance=0.10,
                 traverse_speed=0.30, descent_speed=None,
                 timeout=120.0, stall_seconds=8.0,
                 stall_height=0.02, cross_gain=0.9, max_correction=0.6,
                 sequencer=None, strafe_gain=0.0, max_strafe=0.20,
                 recentre_at=0.15, walkup_speed=None, walkup_slowdown=0.15,
                 flight_margin=0.35):
        self.flight = flight
        # Optional, and the interesting case. Without one the traverse is a
        # steady walk up the flight, which is all a ramp needs. With one, the
        # traverse still owns the steering - staying on the centre line is not
        # something a step-by-step climber can see - and the sequencer owns
        # the throttle: it stops the robot to measure each riser, creeps it up
        # to the face, and steps it over. See go2_voice_control/stepper.py.
        self.sequencer = sequencer
        self.plan = None
        self.direction = direction
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.arrive_radius = arrive_radius
        self.align_tolerance = align_tolerance
        self.traverse_speed = traverse_speed
        # How fast to walk up to a flight after measuring its first riser,
        # while still trotting on the flat floor in front of it. The creep a
        # sequencer asks for is sized for the crawl - it sets the crawl's
        # stride - and on the floor it is only 0.16 m/s of trot: measured,
        # slow enough that the creep timed out 0.4 m into the 0.9 m walk-up
        # and stopped to look again. The trot is steady at twice that and
        # more. It applies only off the flight, going up, and never to the
        # crawl: the robot changes gait at the flight's margin either way.
        # None keeps the walk-up at the sequencer's creep.
        #
        # It hands back to the creep `walkup_slowdown` short of where the gait
        # changes, not at it. Stopping from trot speed carries the robot a few
        # centimetres further, and the crawl wants to start settled well
        # before the riser: measured with the hand-over at the margin itself,
        # the crawl started 2-4 cm closer than it used to, and the one climb
        # out of three that failed had started closest.
        #
        # With the slowdown the crawl starts where it always did, and the walk
        # to the first riser is 12 s shorter - but four climbs went 2 for 4,
        # both failures on the first riser, against 4 for 4 without it. That
        # is too few climbs to call, and too many failures to ship on by
        # default, so the teleop leaves it off unless asked.
        self.walkup_speed = walkup_speed
        self.walkup_slowdown = walkup_slowdown
        # Where the gait changes: Flight.on_flight's margin, as the teleop
        # uses it.
        self.flight_margin = flight_margin
        # Descending is not climbing with the sign flipped. Going up, the
        # break at the foot is concave and the robot walks into it; going
        # down, the break at the head is convex and the trunk pitches
        # nose-first over an edge with nothing under the front feet for a
        # moment. Slower gives the legs time to find the slope.
        self.descent_speed = (descent_speed if descent_speed is not None
                              else traverse_speed * 0.7)
        # How hard to steer back to the centre line, and how far off the
        # flight's own heading that is ever allowed to point. Capped because a
        # robot part-way up a narrow flight that turns 90 degrees to chase the
        # centre line is a robot walking off the side of it.
        self.cross_gain = cross_gain
        self.max_correction = max_correction
        # Correcting sideways instead of by turning. Off by default, because a
        # ramp is wide and a walk up one can afford to steer; a flight is not,
        # and a stepped climb cannot.
        #
        # Two reasons it is the right answer on a staircase. The drift being
        # corrected is itself sideways - the crawl slides across the ground
        # without changing heading, and no amount of yaw authority fixes that
        # quickly: the robot has to turn, walk, and turn back, which on a
        # flight means presenting its feet to the nosings at an angle for the
        # whole manoeuvre. And squareness is worth more here than anywhere
        # else, because a foot that arrives at a riser crabbed catches the
        # corner rather than the face.
        #
        # Measured: with yaw-only correction the robot left the centre line at
        # about a centimetre a second and jammed against the side kerb 0.9 m
        # off centre, one step up, after ninety seconds of trying.
        self.strafe_gain = strafe_gain
        self.max_strafe = max_strafe
        # How far off the centre line is too far to keep climbing through.
        #
        # The cross-track correction is scaled by the sequencer's speed, so
        # it is weakest during a LIFT and switched off entirely during a LOOK
        # - which is to say the robot corrects least while it is climbing,
        # which is when it drifts. Measured: every climb that stayed inside
        # 0.10 m of the line finished, and one that reached 0.40 m walked off
        # the flight. Past this offset the climb stops and sidesteps back
        # onto the line at full authority before going on. A staircase is
        # narrower than the room around it, and there is no hurry that is
        # worth arriving at the next riser crabbed halfway off it.
        self.recentre_at = recentre_at
        self.recentring = False
        self.lateral = 0.0
        self.timeout = timeout
        # A traverse that stops gaining height has failed, and it has to be
        # noticed: a robot scuffing the same nosing forever looks exactly like
        # a robot climbing slowly, and only the height tells them apart.
        self.stall_seconds = stall_seconds
        self.stall_height = stall_height

        self.state = APPROACH
        self.started = None
        self.mark_time = None
        self.mark_height = None
        self.reason = ""

    # ------------------------------------------------------------- helpers

    def _bearing_to(self, pose, point):
        return normalize_angle(math.atan2(point[1] - pose[1],
                                          point[0] - pose[0]) - pose[2])

    def _turn_toward(self, error):
        return 0.0, max(-self.max_angular,
                        min(self.max_angular, 1.2 * error))

    # --------------------------------------------------------------- update

    def update(self, pose, now, profile=None):
        """One step. `pose` is (x, y, yaw, z). Returns (linear, angular).

        `profile` is the latest look at the ground ahead, when there is a
        sequencer to feed it to; see go2_voice_control/perception.py. It is
        optional, and a run without one behaves exactly as it always did.
        """
        if self.started is None:
            self.started = now
            self.mark_time = now
            self.mark_height = pose[3]

        if self.state in (ARRIVED, FAILED):
            self.lateral = 0.0
            return 0.0, 0.0

        if now - self.started > self.timeout:
            self.state = FAILED
            self.lateral = 0.0
            self.reason = "took longer than %.0f s" % self.timeout
            return 0.0, 0.0

        if self.state == APPROACH:
            return self._approach(pose)
        if self.state == ALIGN:
            return self._align(pose, now)
        return self._traverse(pose, now, profile)

    def _approach(self, pose):
        self.lateral = 0.0
        target = self.flight.staging_point(self.direction)
        gap = math.hypot(target[0] - pose[0], target[1] - pose[1])
        if gap <= self.arrive_radius:
            self.state = ALIGN
            return 0.0, 0.0
        error = self._bearing_to(pose, target)
        if abs(error) > 0.5:
            # Turn on the spot rather than driving a long arc into the
            # staircase from the side.
            return self._turn_toward(error)
        speed = self.max_linear * max(0.25, 1.0 - abs(error) / 0.5)
        return speed, max(-self.max_angular,
                          min(self.max_angular, 1.0 * error))

    def _align(self, pose, now):
        self.lateral = 0.0
        error = normalize_angle(
            self.flight.travel_heading(self.direction) - pose[2])
        if abs(error) <= self.align_tolerance:
            self.state = TRAVERSE
            self.mark_time = now
            self.mark_height = pose[3]
            return 0.0, 0.0
        return self._turn_toward(error)

    def _traverse(self, pose, now, profile=None):
        if self.flight.endpoint_reached(pose[:2], pose[3], self.direction):
            self.state = ARRIVED
            self.lateral = 0.0
            return 0.0, 0.0

        if self.sequencer is not None and profile is None:
            # Missing scans are different from a fresh scan of flat ground.
            # Wait for the sensor instead of letting SEARCH walk into a riser
            # on an absent measurement. The overall run timeout still applies.
            self.lateral = 0.0
            self.plan = None
            self.mark_time = now
            self.mark_height = pose[3]
            return 0.0, 0.0

        # The stall clock only runs once the robot is on the flight. Before
        # that it is still walking to the stairs across flat floor, where
        # gaining no height is exactly what should happen - and a stall
        # detector armed through the walk-up reports failure for doing the
        # right thing, the further away the staging point is the sooner.
        if (not self.flight.on_flight(pose)
                or self.flight.height_reached(pose[3], self.direction)):
            # Once at platform height, the remaining task is to bring the
            # rear feet clear of the flight. That level walk cannot gain
            # further height; the overall timeout still bounds it.
            self.mark_time = now
            self.mark_height = pose[3]

        gained = (pose[3] - self.mark_height) * self.direction
        if gained > self.stall_height:
            self.mark_time = now
            self.mark_height = pose[3]
        elif now - self.mark_time > self.stall_seconds:
            self.state = FAILED
            self.lateral = 0.0
            self.reason = ("stopped gaining height %.2f m %s"
                           % (abs(pose[3] - self.flight.bottom_z),
                              "up" if self.direction > 0 else "down"))
            return 0.0, 0.0

        # Steer back onto the centre line, not merely onto the heading.
        # Holding the heading alone keeps whatever sideways offset the robot
        # started with, and a flight is narrower than the room it is in: a
        # descent that began 30 cm off centre walked off the side of a 1.8 m
        # ramp and dropped half a metre onto the floor. It reported success,
        # because it had indeed ended up at the bottom.
        #
        # This is a line-follower: the demanded heading is the flight's, bent
        # toward the centre line in proportion to how far off it the robot is.
        offset = self.flight.cross_track(pose[:2]) * self.direction
        if self.strafe_gain > 0.0:
            # Hold the flight's own heading, and put the whole cross-track
            # correction into a sidestep. Turning with feet split between
            # risers can roll the robot off the edge.
            wanted = self.flight.travel_heading(self.direction)
            sidestep = max(-self.max_strafe,
                           min(self.max_strafe,
                               -self.strafe_gain * offset))
            # The sidestep is commanded in the robot's own frame while the
            # error is measured in the room's, so it only means the same thing
            # while the robot is square to the flight. Faded out by how far
            # off it is, which is the right behaviour rather than a
            # correction: a robot 60 degrees off the flight should be turning
            # back onto it, not crabbing along a line it is not facing.
            self.lateral = sidestep * math.cos(
                normalize_angle(pose[2] - wanted))
            full_lateral = self.lateral
        else:
            self.lateral = 0.0
            full_lateral = 0.0
            correction = max(-self.max_correction,
                             min(self.max_correction,
                                 -self.cross_gain * offset))
            wanted = normalize_angle(
                self.flight.travel_heading(self.direction) + correction)
        error = normalize_angle(wanted - pose[2])
        speed = (self.traverse_speed if self.direction > 0
                 else self.descent_speed)
        angular = max(-self.max_angular, min(self.max_angular, 1.2 * error))

        if self.sequencer is not None:
            self.plan = self.sequencer.update(pose[3], profile, now)
            if self.plan.stalled:
                self.state = FAILED
                self.lateral = 0.0
                self.reason = self.plan.note
                return 0.0, 0.0
            speed *= self.plan.speed
            if (self.walkup_speed is not None and self.direction > 0
                    and self.plan.state in (stepper.MOUNT, stepper.SEARCH)
                    and self.flight.along(pose) < -(self.flight_margin
                                                    + self.walkup_slowdown)):
                speed = max(speed, self.walkup_speed)
            # Nothing to steer with while standing still to look, and a
            # stationary yaw command is not free: the gait turns on the spot
            # for it, which moves the feet the measurement is being taken
            # from and walks the robot off the centre line it was correcting
            # towards in the first place.
            if self.plan.speed <= 0.0:
                angular = 0.0
                self.lateral = 0.0
            else:
                self.lateral *= self.plan.speed
        # Hysteresis: having stopped to re-centre, come most of the way back
        # before climbing again, so the robot does not stall on and off at
        # the threshold for the rest of the flight.
        limit = (self.recentre_at * 0.4 if self.recentring else self.recentre_at)
        if self.strafe_gain > 0.0 and abs(offset) > limit:
            self.recentring = True
            self.lateral = full_lateral
            # The stall detector measures height gained, and a deliberate
            # sidestep gains none. Hold its clock while this is going on.
            self.mark_time = now
            self.mark_height = pose[3]
            return 0.0, angular
        self.recentring = False
        return speed, angular

    @property
    def finished(self):
        return self.state in (ARRIVED, FAILED)

    def describe(self):
        if self.state == ARRIVED:
            climbed = ("" if self.sequencer is None
                       else ", %d steps" % self.sequencer.index)
            return "at the %s%s" % (
                "top" if self.direction > 0 else "bottom", climbed)
        if self.state == FAILED:
            return "gave up: %s" % (self.reason or "unknown")
        if self.state == TRAVERSE and self.plan is not None:
            return "%s: %s" % (TRAVERSE, self.plan.note)
        return self.state


def _main():
    flight = Flight(foot=(2.0, 0.0), top=(4.0, 0.0), bottom_z=0.30,
                    top_z=0.75, heading=0.0)
    failures = 0

    def check(label, ok):
        nonlocal failures
        failures += 0 if ok else 1
        print("%-4s %s" % ("ok" if ok else "FAIL", label))

    up_stage = flight.staging_point(UP)
    check("staging for up is behind the foot", up_stage[0] < flight.foot[0])
    down_stage = flight.staging_point(DOWN)
    check("staging for down is beyond the top", down_stage[0] > flight.top[0])
    check("down travels the reciprocal",
          abs(abs(flight.travel_heading(DOWN)) - math.pi) < 1e-9)

    # Walk the whole errand: start away and facing the wrong way, climb.
    run = StairRun(flight, UP)
    pose = [0.0, 1.5, math.pi, 0.30]
    now = 0.0
    seen = []
    while not run.finished and now < 120.0:
        linear, angular = run.update(tuple(pose), now)
        if run.state not in seen:
            seen.append(run.state)
        pose[2] = normalize_angle(pose[2] + angular * 0.1)
        pose[0] += linear * 0.1 * math.cos(pose[2])
        pose[1] += linear * 0.1 * math.sin(pose[2])
        # The world: height only rises once past the foot of the flight.
        if pose[0] > flight.foot[0]:
            climbed = min(flight.top_z - flight.bottom_z,
                          (pose[0] - flight.foot[0]) * 0.25)
            pose[3] = flight.bottom_z + climbed
        now += 0.1
    check("climbs through approach, align, traverse",
          seen[:3] == [APPROACH, ALIGN, TRAVERSE])
    check("arrives at the top", run.state == ARRIVED)
    print("     %-9s final z=%.2f x=%.2f" % (run.describe(), pose[3], pose[0]))

    # A flight the robot cannot climb has to be given up on, not walked at
    # forever. Standing ON the flight and getting nowhere is what a robot
    # scuffing the same nosing looks like.
    #
    # It has to be started where a traverse actually starts, at the staging
    # point and already square on. Dropping it half way up the flight facing
    # the right way puts it in APPROACH, walking back down to a staging point
    # it can never reach because the pose below never moves - so it fails on
    # the overall clock, and the check passes without ever exercising the
    # stall detector it is named after.
    blocked = StairRun(flight, UP, stall_seconds=3.0)
    pose = list(flight.staging_point(UP)) + [flight.heading, 0.30]
    now = 0.0
    while not blocked.finished and now < 200.0:
        linear, angular = blocked.update(tuple(pose), now)
        pose[2] = normalize_angle(pose[2] + angular * 0.1)
        pose[0] += linear * 0.1 * math.cos(pose[2])
        pose[1] += linear * 0.1 * math.sin(pose[2])
        now += 0.1              # walks on, and never gains a millimetre
    check("gives up on a flight it cannot climb", blocked.state == FAILED)
    check("and gives up on height, not on the clock",
          "height" in blocked.reason)
    print("     %s" % blocked.describe())

    # With a sequencer, the same errand is climbed one measured riser at a
    # time. The scanner is ray cast against the flight rather than simulated,
    # so this is the real perception path with no Gazebo involved.
    from go2_voice_control import perception, stepper

    riser, tread = 0.09, 0.36
    outline = perception.staircase_outline(0.0, riser, tread, 5)
    sequencer = stepper.StepSequencer()
    stepped = StairRun(flight, UP, sequencer=sequencer, traverse_speed=0.30,
                       stall_seconds=25.0)
    pose = [0.0, 1.5, math.pi, 0.30]
    levels, now, dt = {"front": 0, "rear": 0}, 0.0, 0.05
    swings = []
    while not stepped.finished and now < 200.0:
        along = pose[0] - flight.foot[0]
        local = [(px - along, pz - (pose[3] - flight.bottom_z))
                 for px, pz in outline]
        angle_min, increment, ranges = perception.simulate_scan(
            local, 0.30, noise=0.003)
        profile = perception.from_scan(angle_min, increment, ranges)
        linear, angular = stepped.update(tuple(pose), now, profile)
        if stepped.plan is not None:
            swings.append(stepped.plan.swing_height)
        pose[2] = normalize_angle(pose[2] + angular * dt)
        pose[0] += linear * dt * math.cos(pose[2])
        pose[1] += linear * dt * math.sin(pose[2])
        for name, offset in (("front", +0.1934), ("rear", -0.1934)):
            foot = pose[0] - flight.foot[0] + offset
            under = 0
            for index in range(5):
                if foot >= index * tread:
                    under = index + 1
            if under > levels[name]:
                if stepped.plan and stepped.plan.swing_height >= riser + 0.004:
                    levels[name] += 1
            elif under < levels[name]:
                levels[name] = under
        wanted = flight.bottom_z + riser * (levels["front"] + levels["rear"]) / 2.0
        pose[3] += (wanted - pose[3]) * min(1.0, dt / 0.35)
        now += dt
    check("a stepped run gets to the top too", stepped.state == ARRIVED)
    check("having climbed the steps one at a time", sequencer.index >= 4)
    check("on a foot clearance it measured, not one it was given",
          abs(max(swings) - (riser + sequencer.clearance)) < 0.02)
    print("     %s, swing %.3f m, %.0f s" % (stepped.describe(),
                                             max(swings), now))

    print("\n%s" % ("all checks passed" if not failures
                    else "%d CHECKS FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
