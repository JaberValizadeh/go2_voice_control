#!/usr/bin/env python3
"""Stair command wiring using real ROS messages, without a ROS master."""

import importlib.util
import os
import unittest
from unittest import mock

import rospy
from std_msgs.msg import Float64

from go2_voice_control import perception, stairs, stepper


SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "voice_teleop.py")
SPEC = importlib.util.spec_from_file_location("voice_teleop", SCRIPT)
teleop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teleop)


class TestStairWiring(unittest.TestCase):

    def setUp(self):
        self.now = 0.0
        self.params = {"~stair_foot": [2.0, 0.0], "~stair_top": [4.0, 0.0]}
        for name, replacement in (
                ("get_param", lambda key, default=None: self.params.get(key, default)),
                ("Publisher", mock.Mock()), ("Subscriber", mock.Mock()),
                ("loginfo", mock.Mock()), ("logwarn", mock.Mock()),
                ("loginfo_throttle", mock.Mock())):
            patch = mock.patch.object(teleop.rospy, name, replacement)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(rospy.Time, "now",
                                  side_effect=lambda: rospy.Time.from_sec(self.now))
        patch.start()
        self.addCleanup(patch.stop)
        self.node = teleop.VoiceTeleop()
        self.node.position = self.node.flight.staging_point(stairs.UP)
        self.node.yaw = 0.0
        self.node.height = 0.30

    def scan(self, edge=0.60):
        self.node.profile = perception.Profile(
            -0.30, [perception.Step(edge=edge, rise=0.09, top=0.09)])
        self.node.scan_stamp = rospy.Time.from_sec(self.now)

    def tick(self):
        self.now += 0.1
        self.node.update_stairs()

    def test_a_late_scanner_is_used_without_repeating_the_command(self):
        self.node.start_stairs(stairs.UP)
        self.assertIsNotNone(self.node.stair_run.sequencer)
        for _ in range(3):
            self.tick()
        self.assertEqual(self.node.stair_run.state, stairs.TRAVERSE)
        self.assertEqual(self.node.twist.linear.x, 0.0)
        for _ in range(6):
            self.scan()
            self.tick()
        self.assertEqual(self.node.stair_run.sequencer.state, stepper.MOUNT)
        self.assertGreater(self.node.twist.linear.x, 0.0)

    def test_scanner_dropout_stops_an_active_traverse(self):
        self.scan()
        self.node.start_stairs(stairs.UP)
        self.node.stair_run.state = stairs.TRAVERSE
        for _ in range(6):
            self.scan()
            self.tick()
        self.assertGreater(self.node.twist.linear.x, 0.0)
        self.now += 1.1
        self.tick()
        self.assertEqual(self.node.twist.linear.x, 0.0)
        self.assertEqual(self.node.twist.linear.y, 0.0)
        self.assertEqual(self.node.twist.angular.z, 0.0)
        self.assertIsNotNone(self.node.stair_run)

    def test_a_new_stair_command_restores_the_approach_gait(self):
        self.node.start_stairs(stairs.UP)
        self.node.set_stair_mode(True)
        self.node.start_stairs(stairs.DOWN)
        self.assertFalse(self.node.stair_mode)
        self.assertEqual(self.node.stair_run.state, stairs.APPROACH)
        self.assertEqual(self.node.stair_run.direction, stairs.DOWN)

    def test_climbing_mode_engages_on_the_first_lift_tick(self):
        self.scan(edge=0.40)
        self.node.start_stairs(stairs.UP)
        run = self.node.stair_run
        run.state = stairs.TRAVERSE
        self.node.position = (1.60, 0.0)
        self.assertFalse(run.flight.on_flight(self.node.position))
        run.sequencer.state = stepper.MOUNT
        run.sequencer.riser = 0.09
        run.sequencer.swing_height = 0.135
        self.tick()
        self.assertEqual(run.sequencer.state, stepper.LIFT)
        self.assertTrue(self.node.stair_mode)
        self.assertGreater(self.node.twist.linear.x, 0.0)

    def test_explicitly_disabling_the_scanner_keeps_the_ramp_traverse(self):
        self.node.use_scanner = False
        self.node.start_stairs(stairs.UP)
        self.assertIsNone(self.node.stair_run.sequencer)
        self.node.stair_run.state = stairs.TRAVERSE
        self.tick()
        self.assertGreater(self.node.twist.linear.x, 0.0)

    def test_crouching_does_not_erase_a_climbed_riser(self):
        self.scan(edge=0.40)
        self.node.start_stairs(stairs.UP)
        run = self.node.stair_run
        run.state = stairs.TRAVERSE
        self.node.position = (2.0, 0.0)
        run.sequencer.state = stepper.MOUNT
        run.sequencer.riser = 0.09
        self.tick()
        self.node.height = 0.25
        self.node.on_height_offset(Float64(0.05))
        self.tick()
        self.assertEqual(run.sequencer.index, 0)
        self.node.height += 0.09
        self.tick()
        self.assertEqual(run.sequencer.index, 1)

    def test_the_crouched_robot_can_finish_on_the_platform(self):
        self.node.start_stairs(stairs.UP)
        self.node.stair_run.state = stairs.TRAVERSE
        self.node.position = (4.2, 0.0)
        self.node.height = 0.68
        self.node.on_height_offset(Float64(0.05))
        self.tick()
        self.assertIsNone(self.node.stair_run)
        self.assertFalse(self.node.stair_mode)

    def test_the_stair_timeout_is_configurable(self):
        self.params["~stair_timeout"] = 350.0
        self.node.start_stairs(stairs.UP)
        self.assertEqual(self.node.stair_run.timeout, 350.0)

    def test_the_step_timer_allows_the_supported_crawl_to_raise_the_trunk(self):
        self.node.start_stairs(stairs.UP)
        self.assertEqual(self.node.stair_run.sequencer.lift_timeout, 30.0)
        self.params["~stair_lift_timeout"] = 40.0
        self.node.start_stairs(stairs.UP)
        self.assertEqual(self.node.stair_run.sequencer.lift_timeout, 40.0)


if __name__ == "__main__":
    unittest.main()
