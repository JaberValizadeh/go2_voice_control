#!/usr/bin/env python3
"""Integration tests: drive the real voice_teleop node over ROS topics.

The node is fed a synthetic /gazebo/model_states so the tests control the
robot's heading directly and a turn can be completed in a fraction of a second
instead of the ~14 s it takes in simulation.
"""

import math
import sys
import threading
import time
import unittest

import rospy
import rostest
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import String

PKG = "go2_voice_control"
NAME = "test_voice_teleop"

MODEL = "unitree_go2"


def yaw_pose(yaw):
    pose = Pose()
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)
    return pose


class TestVoiceTeleop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rospy.init_node(NAME, anonymous=True)
        cls.commands = rospy.Publisher("/voice_commands", String,
                                       queue_size=10)
        cls.states = rospy.Publisher("/gazebo/model_states", ModelStates,
                                     queue_size=1)
        cls.latest = None
        cls.lock = threading.Lock()

        def on_cmd_vel(message):
            with cls.lock:
                cls.latest = message

        rospy.Subscriber("/cmd_vel", Twist, on_cmd_vel, queue_size=1)

        cls.yaw = 0.0
        cls.running = True

        def publish_pose():
            rate = rospy.Rate(50)
            while cls.running and not rospy.is_shutdown():
                message = ModelStates()
                message.name = ["ground_plane", MODEL]
                message.pose = [Pose(), yaw_pose(cls.yaw)]
                message.twist = [Twist(), Twist()]
                cls.states.publish(message)
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break

        cls.thread = threading.Thread(target=publish_pose)
        cls.thread.daemon = True
        cls.thread.start()

        # Wait for the node under test to connect on both sides.
        deadline = time.time() + 20
        while time.time() < deadline:
            if (cls.commands.get_num_connections() > 0
                    and cls.latest is not None):
                break
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.running = False

    # ---------------------------------------------------------------- helpers

    def cmd_vel(self):
        with self.lock:
            return self.latest

    def say(self, text, settle=0.6):
        type(self).yaw = 0.0
        self.commands.publish(String(data=text))
        time.sleep(settle)

    def wait_for(self, predicate, timeout=8.0, spin_yaw=None):
        """Wait until predicate(cmd_vel) holds, optionally driving the yaw."""
        deadline = time.time() + timeout
        start = time.time()
        while time.time() < deadline:
            if spin_yaw is not None:
                type(self).yaw = spin_yaw(time.time() - start)
            message = self.cmd_vel()
            if message is not None and predicate(message):
                return message
            time.sleep(0.02)
        return None

    # ------------------------------------------------------------------ tests

    def test_node_publishes_cmd_vel(self):
        self.assertIsNotNone(self.cmd_vel(),
                             "voice_teleop is not publishing /cmd_vel")

    def test_forward_then_stop(self):
        self.say("move forward")
        message = self.cmd_vel()
        self.assertGreater(message.linear.x, 0.0)
        self.assertAlmostEqual(message.angular.z, 0.0, places=6)

        self.say("stop")
        message = self.cmd_vel()
        self.assertAlmostEqual(message.linear.x, 0.0, places=6)
        self.assertAlmostEqual(message.angular.z, 0.0, places=6)

    def test_backward_walks_in_reverse(self):
        """'move backward' must reverse, not spin 180 degrees."""
        self.say("move backward")
        message = self.cmd_vel()
        self.assertLess(message.linear.x, 0.0,
                        "backward should command negative linear velocity")
        self.assertAlmostEqual(message.angular.z, 0.0, places=6)
        self.say("stop")

    def test_turn_left_and_right_have_opposite_signs(self):
        self.say("turn left")
        left = self.cmd_vel().angular.z
        self.assertGreater(left, 0.0)
        self.say("stop")

        self.say("turn right")
        right = self.cmd_vel().angular.z
        self.assertLess(right, 0.0)
        self.say("stop")

    def test_turn_completes_when_heading_reached(self):
        self.say("turn left", settle=0.3)
        self.assertGreater(self.cmd_vel().angular.z, 0.0)
        # Sweep the heading past the 90 degree target.
        done = self.wait_for(lambda m: abs(m.angular.z) < 1e-6,
                             timeout=6.0,
                             spin_yaw=lambda t: min(math.pi / 2.0, t * 1.2))
        self.assertIsNotNone(done, "turn never finished")
        self.say("stop")

    def test_go_back_drives_forward_after_turning(self):
        """Regression: the drive after a half turn was cancelled instantly.

        A half turn takes longer than command_timeout, and the timeout was
        measured from the original utterance, so the forward motion the turn
        was meant to lead into was zeroed on the cycle it began.
        """
        self.say("go back", settle=0.3)
        self.assertGreater(abs(self.cmd_vel().angular.z), 0.0,
                           "go back should start turning")

        # The half turn must take longer than command_timeout (2 s in the
        # test launch file), otherwise the timeout never expires during the
        # turn and the regression cannot show itself. At 1.0 rad/s the sweep
        # takes about 3.1 s.
        moving = self.wait_for(lambda m: m.linear.x > 0.0,
                               timeout=12.0,
                               spin_yaw=lambda t: min(math.pi, t * 1.0))
        self.assertIsNotNone(
            moving, "robot did not drive forward after turning around")
        self.say("stop")

    def test_faster_and_slower_change_speed(self):
        self.say("move forward")
        base = self.cmd_vel().linear.x

        self.commands.publish(String(data="faster"))
        time.sleep(0.6)
        faster = self.cmd_vel().linear.x
        self.assertGreater(faster, base)

        self.commands.publish(String(data="slower"))
        time.sleep(0.6)
        self.assertLess(self.cmd_vel().linear.x, faster)
        self.say("stop")

    def test_speed_is_capped(self):
        self.say("move forward")
        for _ in range(10):
            self.commands.publish(String(data="faster"))
            time.sleep(0.15)
        time.sleep(0.5)
        # max_linear_speed is 0.4 in the test launch file.
        self.assertLessEqual(self.cmd_vel().linear.x, 0.4 + 1e-6)
        self.say("stop")

    def test_unknown_command_does_not_change_motion(self):
        self.say("move forward")
        before = self.cmd_vel().linear.x
        self.commands.publish(String(data="make me a sandwich"))
        time.sleep(0.6)
        self.assertAlmostEqual(self.cmd_vel().linear.x, before, places=6)
        self.say("stop")

    def test_command_timeout_stops_the_robot(self):
        """Walking must stop by itself when no new command is heard."""
        self.say("move forward")
        self.assertGreater(self.cmd_vel().linear.x, 0.0)
        stopped = self.wait_for(lambda m: abs(m.linear.x) < 1e-6, timeout=8.0)
        self.assertIsNotNone(stopped, "command_timeout never stopped the robot")


if __name__ == "__main__":
    rostest.rosrun(PKG, NAME, TestVoiceTeleop, sys.argv)
