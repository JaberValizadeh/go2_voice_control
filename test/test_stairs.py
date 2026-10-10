#!/usr/bin/env python3
"""The stair errand: approach, align, traverse, stop.

None of this needs ROS or a simulator - the state machine takes a pose and
answers with a velocity, so a whole climb can be flown in a loop.
"""

import math
import unittest

from go2_voice_control import stairs, stepper


def flight(**kwargs):
    settings = dict(foot=(2.0, 0.0), top=(4.0, 0.0), bottom_z=0.30,
                    top_z=0.75, heading=0.0)
    settings.update(kwargs)
    return stairs.Flight(**settings)


def fly(run, start, world, limit=200.0, step=0.1):
    """Run the machine against a toy world. Returns the states it went through.

    `world` maps a position to the height the robot would be at there, which
    is what makes this a test of the behaviour rather than of arithmetic.
    """
    pose = list(start)
    now = 0.0
    seen = []
    while not run.finished and now < limit:
        linear, angular = run.update(tuple(pose), now)
        if not seen or seen[-1] != run.state:
            seen.append(run.state)
        pose[2] = stairs.normalize_angle(pose[2] + angular * step)
        pose[0] += linear * step * math.cos(pose[2])
        pose[1] += linear * step * math.sin(pose[2])
        pose[3] = world(pose[0], pose[1])
        now += step
    return seen, pose, now


def slope(low=0.30, high=0.75, foot=2.0, top=4.0):
    """A world that rises linearly between the foot and the top."""
    def height(x, _y):
        if x <= foot:
            return low
        if x >= top:
            return high
        return low + (high - low) * (x - foot) / (top - foot)
    return height


class TestStagingPoint(unittest.TestCase):
    """Where the robot stands before it sets off.

    Walking straight at a flight from wherever it happens to be is what makes
    a robot mount it diagonally and catch a foot on the edge. Approaching a
    fixed point first means the climb always starts square on.
    """

    def test_going_up_stages_below_the_foot(self):
        point = flight().staging_point(stairs.UP)
        self.assertLess(point[0], 2.0)

    def test_going_down_stages_beyond_the_top(self):
        point = flight().staging_point(stairs.DOWN)
        self.assertGreater(point[0], 4.0)

    def test_the_stage_distance_is_respected(self):
        point = flight(stage=2.5).staging_point(stairs.UP)
        self.assertAlmostEqual(point[0], -0.5, places=9)

    def test_a_flight_facing_north_stages_to_the_south(self):
        sideways = flight(foot=(0.0, 2.0), top=(0.0, 4.0),
                          heading=math.pi / 2.0)
        point = sideways.staging_point(stairs.UP)
        self.assertAlmostEqual(point[0], 0.0, places=6)
        self.assertLess(point[1], 2.0)


class TestTravelHeading(unittest.TestCase):

    def test_up_is_the_flight_heading(self):
        self.assertAlmostEqual(flight().travel_heading(stairs.UP), 0.0,
                               places=9)

    def test_down_is_the_reciprocal(self):
        self.assertAlmostEqual(abs(flight().travel_heading(stairs.DOWN)),
                               math.pi, places=9)


class TestHeightIsTheTest(unittest.TestCase):
    """A traverse ends on height, not on distance.

    Distance would mean trusting that the robot walked as far as it was told,
    and on a slope it does not - the feet slip and the body pitches. "Am I
    45 cm higher than I was" is the question actually being asked.
    """

    def test_reaching_the_top_ends_an_upward_run(self):
        self.assertTrue(flight().height_reached(0.74, stairs.UP))
        self.assertFalse(flight().height_reached(0.55, stairs.UP))

    def test_reaching_the_bottom_ends_a_downward_run(self):
        self.assertTrue(flight().height_reached(0.32, stairs.DOWN))
        self.assertFalse(flight().height_reached(0.60, stairs.DOWN))

    def test_the_tolerance_is_generous_enough_for_a_standing_robot(self):
        """The trunk bobs a few centimetres while walking; the test must not
        need the robot to stop moving to notice it has arrived."""
        self.assertTrue(flight().height_reached(0.70, stairs.UP,
                                                tolerance=0.06))

    def test_top_completion_leaves_the_rear_feet_off_the_slope(self):
        self.assertFalse(flight().endpoint_reached((3.8, 0.0), 0.70,
                                                   stairs.UP))
        self.assertTrue(flight().endpoint_reached((4.2, 0.0), 0.75,
                                                  stairs.UP))

    def test_bottom_completion_clears_the_foot_of_the_slope(self):
        self.assertFalse(flight().endpoint_reached((2.1, 0.0), 0.32,
                                                   stairs.DOWN))
        self.assertTrue(flight().endpoint_reached((1.8, 0.0), 0.30,
                                                  stairs.DOWN))


class TestAWholeClimb(unittest.TestCase):

    def test_it_approaches_aligns_and_traverses_in_that_order(self):
        run = stairs.StairRun(flight(), stairs.UP)
        seen, pose, _ = fly(run, (0.0, 1.5, math.pi, 0.30), slope())
        self.assertEqual(seen[:3], [stairs.APPROACH, stairs.ALIGN,
                                    stairs.TRAVERSE])
        self.assertEqual(run.state, stairs.ARRIVED)
        self.assertGreater(pose[3], 0.69)

    def test_it_climbs_from_behind_and_facing_away(self):
        """The hard start: the robot is past the flight and pointing off it."""
        run = stairs.StairRun(flight(), stairs.UP)
        seen, pose, _ = fly(run, (-2.0, -2.0, -math.pi / 2.0, 0.30), slope())
        self.assertEqual(run.state, stairs.ARRIVED, seen)

    def test_it_comes_back_down(self):
        run = stairs.StairRun(flight(), stairs.DOWN)
        seen, pose, _ = fly(run, (5.5, 0.4, 0.0, 0.75), slope())
        self.assertEqual(run.state, stairs.ARRIVED, seen)
        self.assertLess(pose[3], 0.37)

    def test_a_run_that_is_already_there_still_terminates(self):
        run = stairs.StairRun(flight(), stairs.UP)
        _, _, _ = fly(run, (4.5, 0.0, 0.0, 0.75), slope())
        self.assertEqual(run.state, stairs.ARRIVED)


class TestGivingUp(unittest.TestCase):
    """A robot that cannot climb has to say so, not shuffle forever."""

    def test_finishing_at_platform_height_does_not_require_more_height_gain(self):
        run = stairs.StairRun(flight(), stairs.UP, stall_seconds=1.0)
        run.state = stairs.TRAVERSE
        run.started = run.mark_time = 0.0
        run.mark_height = 0.74
        for tick in range(200):
            run.update((4.0, 0.0, 0.0, 0.74), tick * 0.1)
        self.assertEqual(run.state, stairs.TRAVERSE)
        run.update((4.21, 0.0, 0.0, 0.74), 20.0)
        self.assertEqual(run.state, stairs.ARRIVED)

    def test_platform_exit_is_still_bounded_by_the_overall_timeout(self):
        run = stairs.StairRun(flight(), stairs.UP, timeout=5.0, stall_seconds=1.0)
        run.state = stairs.TRAVERSE
        run.started = run.mark_time = 0.0
        run.mark_height = 0.74
        run.update((4.0, 0.0, 0.0, 0.74), 6.0)
        self.assertEqual(run.state, stairs.FAILED)
        self.assertIn("longer than", run.reason)

    def test_a_flight_it_cannot_climb_is_abandoned(self):
        run = stairs.StairRun(flight(), stairs.UP, stall_seconds=3.0)
        flat = lambda x, y: 0.30           # noqa: E731 - never rises
        _, _, elapsed = fly(run, (0.0, 0.0, 0.0, 0.30), flat, limit=120.0)
        self.assertEqual(run.state, stairs.FAILED)
        self.assertIn("height", run.reason)
        self.assertLess(elapsed, 60.0, "should give up long before the timeout")

    def test_the_stall_clock_does_not_run_during_the_walk_up(self):
        """Before the first step there is no height to gain, and a stall
        detector armed through the approach reports failure for doing exactly
        the right thing - the further away the staging point, the sooner."""
        run = stairs.StairRun(flight(), stairs.UP, stall_seconds=1.0)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        # Well short of the foot, flat floor, for far longer than the window.
        for tick in range(200):
            run.update((0.0, 0.0, 0.0, 0.30), tick * 0.1)
        self.assertEqual(run.state, stairs.TRAVERSE)

    def test_but_it_does_run_once_on_the_flight(self):
        run = stairs.StairRun(flight(), stairs.UP, stall_seconds=1.0)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        for tick in range(200):
            run.update((2.3, 0.0, 0.0, 0.30), tick * 0.1)
        self.assertEqual(run.state, stairs.FAILED)
        self.assertIn("height", run.reason)

    def test_slow_progress_is_not_giving_up(self):
        """Climbing slowly and being stuck look identical for a few seconds,
        and only the height separates them."""
        run = stairs.StairRun(flight(), stairs.UP, stall_seconds=3.0)
        crawl = slope()
        seen, _, _ = fly(run, (1.9, 0.0, 0.0, 0.30), crawl)
        self.assertEqual(run.state, stairs.ARRIVED, seen)

    def test_the_overall_timeout_still_applies(self):
        run = stairs.StairRun(flight(), stairs.UP, timeout=5.0,
                              stall_seconds=99.0)
        # Never arrives: the robot is held away from the flight.
        held = lambda x, y: 0.30          # noqa: E731
        fly(run, (-30.0, 0.0, 0.0, 0.30), held, limit=60.0)
        self.assertEqual(run.state, stairs.FAILED)
        self.assertIn("longer than", run.reason)

    def test_a_finished_run_commands_nothing(self):
        run = stairs.StairRun(flight(), stairs.UP)
        fly(run, (4.5, 0.0, 0.0, 0.75), slope())
        self.assertEqual(run.update((4.5, 0.0, 0.0, 0.75), 999.0), (0.0, 0.0))


class TestOnFlight(unittest.TestCase):

    def test_before_the_foot_is_not_on_the_flight(self):
        self.assertFalse(flight().on_flight((0.0, 0.0)))

    def test_between_foot_and_top_is(self):
        self.assertTrue(flight().on_flight((3.0, 0.0)))

    def test_well_past_the_top_is_not(self):
        self.assertFalse(flight().on_flight((6.0, 0.0)))

    def test_a_little_past_either_end_still_counts(self):
        """The robot's feet reach beyond its trunk, so the margin matters."""
        self.assertTrue(flight().on_flight((1.8, 0.0)))
        self.assertTrue(flight().on_flight((4.2, 0.0)))


class TestDescentIsNotClimbingBackwards(unittest.TestCase):

    def test_descent_is_slower_by_default(self):
        """Going up, the break at the foot is concave and the robot walks
        into it. Going down, the break at the head is convex and the trunk
        pitches over an edge with nothing under the front feet."""
        run = stairs.StairRun(flight(), stairs.DOWN, traverse_speed=0.30)
        self.assertLess(run.descent_speed, 0.30)

    def test_it_can_be_set_explicitly(self):
        run = stairs.StairRun(flight(), stairs.DOWN, traverse_speed=0.30,
                              descent_speed=0.12)
        self.assertAlmostEqual(run.descent_speed, 0.12, places=9)

    def test_going_up_uses_the_traverse_speed(self):
        run = stairs.StairRun(flight(), stairs.UP, traverse_speed=0.30,
                              descent_speed=0.05)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        linear, _ = run.update((3.0, 0.0, 0.0, 0.45), 1.0)
        self.assertAlmostEqual(linear, 0.30, places=9)

    def test_going_down_uses_the_descent_speed(self):
        run = stairs.StairRun(flight(), stairs.DOWN, traverse_speed=0.30,
                              descent_speed=0.05)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.75
        linear, _ = run.update((3.0, 0.0, math.pi, 0.60), 1.0)
        self.assertAlmostEqual(linear, 0.05, places=9)


class TestCrossTrack(unittest.TestCase):
    """Staying on the flight, not merely pointing along it."""

    def test_on_the_centre_line_is_zero(self):
        self.assertAlmostEqual(flight().cross_track((3.0, 0.0)), 0.0,
                               places=9)

    def test_left_of_the_line_is_positive(self):
        self.assertGreater(flight().cross_track((3.0, 0.5)), 0.0)

    def test_it_works_for_a_flight_at_an_angle(self):
        diagonal = stairs.Flight(foot=(0.0, 0.0), top=(2.0, 2.0),
                                 bottom_z=0.3, top_z=0.75,
                                 heading=math.pi / 4.0)
        self.assertAlmostEqual(diagonal.cross_track((1.0, 1.0)), 0.0,
                               places=9)
        self.assertGreater(diagonal.cross_track((0.0, 1.0)), 0.0)


class TestItHoldsItsLine(unittest.TestCase):

    def test_strafe_mode_corrects_without_turning_on_the_steps(self):
        run = stairs.StairRun(flight(), stairs.UP, strafe_gain=0.9,
                              max_strafe=0.2)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        _, angular = run.update((3.0, -0.5, 0.0, 0.45), 1.0)
        self.assertGreater(run.lateral, 0.0)
        self.assertAlmostEqual(angular, 0.0)

    def test_descent_strafe_reduces_the_world_frame_offset(self):
        for heading in (0.0, math.pi / 2.0):
            for offset in (-0.5, 0.5):
                with self.subTest(heading=heading, offset=offset):
                    line = flight(foot=(0.0, 0.0),
                                  top=(2.0 * math.cos(heading),
                                       2.0 * math.sin(heading)),
                                  heading=heading)
                    run = stairs.StairRun(line, stairs.DOWN, strafe_gain=0.9)
                    run.state = stairs.TRAVERSE
                    yaw = line.travel_heading(stairs.DOWN)
                    x = math.cos(heading) - offset * math.sin(heading)
                    y = math.sin(heading) + offset * math.cos(heading)
                    linear, angular = run.update((x, y, yaw, 0.60), 0.0)
                    moved = (x + 0.1 * (linear * math.cos(yaw)
                                       - run.lateral * math.sin(yaw)),
                             y + 0.1 * (linear * math.sin(yaw)
                                       + run.lateral * math.cos(yaw)))
                    self.assertLess(abs(line.cross_track(moved)), abs(offset))
                    self.assertAlmostEqual(angular, 0.0)

    def test_drifting_off_the_side_is_steered_back(self):
        """The failure this exists for: a descent that began 30 cm off
        centre walked off the side of a 1.8 m ramp and dropped half a metre,
        then reported success because it had indeed reached the bottom."""
        run = stairs.StairRun(flight(), stairs.UP)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        # On the flight, pointing exactly along it, but half a metre left.
        _, angular = run.update((3.0, 0.5, 0.0, 0.45), 1.0)
        self.assertLess(angular, 0.0, "off to the left, should steer right")
        _, angular = run.update((3.0, -0.5, 0.0, 0.45), 1.0)
        self.assertGreater(angular, 0.0)

    def test_the_correction_is_capped(self):
        """A robot part-way up a narrow flight that turns 90 degrees to chase
        the centre line is a robot walking off the side of it."""
        run = stairs.StairRun(flight(), stairs.UP, max_correction=0.4)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        # Absurdly far off; the demanded heading must still be sane.
        _, angular = run.update((3.0, 20.0, 0.0, 0.45), 1.0)
        self.assertGreaterEqual(angular, -1.2 * 0.4 - 1e-9)

    def test_going_down_corrects_the_same_way(self):
        """The cross-track sign has to flip with the direction of travel, or
        the correction drives the robot off the edge it was avoiding."""
        run = stairs.StairRun(flight(), stairs.DOWN)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.75
        # Travelling in -x; being at y=+0.5 is off to the robot's RIGHT.
        _, angular = run.update((3.0, 0.5, math.pi, 0.60), 1.0)
        self.assertGreater(angular, 0.0)

    def test_the_traverse_steers_back_onto_the_heading(self):
        """A flight is narrower than the room, and a metre of drift puts a
        foot off the side of it."""
        run = stairs.StairRun(flight(), stairs.UP)
        run.state = stairs.TRAVERSE
        run.started = 0.0
        run.mark_time = 0.0
        run.mark_height = 0.30
        _, angular = run.update((2.5, 0.0, 0.3, 0.40), 1.0)
        self.assertLess(angular, 0.0, "yawed left, should steer right")
        _, angular = run.update((2.5, 0.0, -0.3, 0.40), 1.0)
        self.assertGreater(angular, 0.0)

    def test_it_turns_on_the_spot_rather_than_arcing_in_from_the_side(self):
        run = stairs.StairRun(flight(), stairs.UP)
        linear, angular = run.update((0.0, 3.0, 0.0, 0.30), 0.0)
        self.assertEqual(linear, 0.0)
        self.assertNotEqual(angular, 0.0)


class TestTraverseStops(unittest.TestCase):

    def run_with_lateral_motion(self, **kwargs):
        run = stairs.StairRun(flight(), stairs.UP, strafe_gain=0.9, **kwargs)
        run.state = stairs.TRAVERSE
        run.update((3.0, 0.2, 0.0, 0.45), 0.0)
        self.assertNotEqual(run.lateral, 0.0)
        return run

    def test_arrival_clears_lateral_motion_immediately(self):
        run = self.run_with_lateral_motion()
        self.assertEqual(run.update((4.2, 0.2, 0.0, 0.75), 0.1), (0.0, 0.0))
        self.assertEqual(run.lateral, 0.0)

    def test_timeout_clears_lateral_motion_immediately(self):
        run = self.run_with_lateral_motion(timeout=1.0)
        self.assertEqual(run.update((3.0, 0.2, 0.0, 0.45), 1.1), (0.0, 0.0))
        self.assertEqual(run.lateral, 0.0)

    def test_height_stall_clears_lateral_motion_immediately(self):
        run = self.run_with_lateral_motion(stall_seconds=1.0)
        self.assertEqual(run.update((3.0, 0.2, 0.0, 0.45), 1.1), (0.0, 0.0))
        self.assertEqual(run.lateral, 0.0)

    def test_missing_scans_hold_the_traverse_until_the_run_times_out(self):
        run = stairs.StairRun(flight(), stairs.UP,
                              sequencer=stepper.StepSequencer(), timeout=3.0,
                              stall_seconds=0.5, strafe_gain=0.9)
        run.state = stairs.TRAVERSE
        pose = (2.0, 0.2, 0.0, 0.30)
        for tick in range(30):
            self.assertEqual(run.update(pose, tick * 0.1), (0.0, 0.0))
            self.assertEqual(run.lateral, 0.0)
            self.assertEqual(run.state, stairs.TRAVERSE)
        run.update(pose, 3.1)
        self.assertEqual(run.state, stairs.FAILED)
        self.assertIn("longer than", run.reason)


class FixedPlan(object):
    """A sequencer that always wants the same thing."""

    def __init__(self, state, speed):
        self.state, self.speed = state, speed

    def update(self, z, profile, now):
        return stepper.Plan(self.speed, 0.06, self.state)


class TestWalkUp(unittest.TestCase):
    """The flat walk-up to a flight goes at trot speed, not crawl creep."""

    def traverse(self, state, x, direction=stairs.UP, walkup=0.30):
        run = stairs.StairRun(flight(), direction,
                              sequencer=FixedPlan(state, 0.54),
                              walkup_speed=walkup)
        run.state = stairs.TRAVERSE
        heading = 0.0 if direction > 0 else math.pi
        z = 0.30 if direction > 0 else 0.75
        linear, _ = run.update((x, 0.0, heading, z), 0.0, profile=object())
        return linear

    def test_creeping_to_the_first_riser_is_at_walk_up_speed(self):
        self.assertAlmostEqual(self.traverse(stepper.MOUNT, 0.9), 0.30)

    def test_on_the_flight_it_is_the_crawl_creep_again(self):
        self.assertAlmostEqual(self.traverse(stepper.MOUNT, 2.2), 0.30 * 0.54)

    def test_it_slows_before_the_gait_changes_not_at_it(self):
        # The foot is at 2.0 and the gait changes 0.35 m before it; the trot
        # hands back to the creep 0.15 m before that.
        self.assertAlmostEqual(self.traverse(stepper.MOUNT, 1.45), 0.30)
        self.assertAlmostEqual(self.traverse(stepper.MOUNT, 1.55), 0.30 * 0.54)

    def test_stepping_up_is_never_hurried(self):
        self.assertAlmostEqual(self.traverse(stepper.LIFT, 0.9), 0.30 * 0.54)

    def test_standing_to_look_stays_standing(self):
        run = stairs.StairRun(flight(), stairs.UP,
                              sequencer=FixedPlan(stepper.LOOK, 0.0),
                              walkup_speed=0.30)
        run.state = stairs.TRAVERSE
        linear, _ = run.update((0.9, 0.0, 0.0, 0.30), 0.0, profile=object())
        self.assertEqual(linear, 0.0)

    def test_going_down_is_unchanged(self):
        self.assertLess(self.traverse(stepper.MOUNT, 5.0, stairs.DOWN), 0.30)

    def test_off_by_default(self):
        self.assertAlmostEqual(self.traverse(stepper.MOUNT, 0.9, walkup=None),
                               0.30 * 0.54)


if __name__ == "__main__":
    unittest.main()
