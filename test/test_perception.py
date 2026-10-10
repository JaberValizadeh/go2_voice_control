#!/usr/bin/env python3
"""Seeing a staircase: what the scanner reports, and what it must not.

Every scan here is ray cast against a staircase whose riser height is known
exactly, so these are tests of the measurement and not of the arithmetic that
happens to be in front of it. No ROS, no Gazebo.
"""

import math
import unittest

from go2_voice_control import perception


RISER, TREAD = 0.09, 0.36
RIDE = 0.30                       # trunk height above the tread it stands on


def flight(foot=0.70, riser=RISER, tread=TREAD, count=5):
    return perception.staircase_outline(foot, riser, tread, count)


def look(outline, ride=RIDE, pitch=0.0, noise=0.003, **kwargs):
    angle_min, increment, ranges = perception.simulate_scan(
        outline, ride, pitch=pitch, noise=noise)
    return perception.from_scan(angle_min, increment, ranges, pitch=pitch,
                                **kwargs)


def descent(edge=0.55, riser=RISER, tread=TREAD, count=5):
    """A flight seen from the top: the ground drops away instead of rising."""
    outline = [(-1.0, 0.0), (edge, 0.0)]
    for index in range(count):
        top = -riser * (index + 1)
        outline.append((edge + tread * index, top))
        outline.append((edge + tread * (index + 1), top))
    outline.append((outline[-1][0] + 6.0, outline[-1][1]))
    return outline


class TestGround(unittest.TestCase):
    """Finding the surface the robot is standing on.

    Everything else is measured relative to it, so getting it wrong by one
    riser makes a flight of even steps look like flat ground.
    """

    def test_foothold_boundary_excludes_riser_rays_merged_into_a_tread(self):
        profile = perception.Profile(0.27, [], treads=[(2.95, 3.38, 0.27)],
            points=[(2.95, 0.270), (3.05, 0.271), (3.20, 0.269),
                    (3.278, 0.271), (3.281, 0.31), (3.30, 0.359), (3.38, 0.36)])
        self.assertEqual(profile.footholds(), [(2.95, 3.278, 0.27)])

    def test_an_intermediate_height_without_a_flat_surface_is_not_a_foothold(self):
        profile = perception.Profile(0.42, [], treads=[(3.5, 3.9, 0.42)],
            points=[(3.5, 0.36), (3.6, 0.36), (3.64, 0.42),
                    (3.7, 0.45), (3.9, 0.45)])
        self.assertEqual(profile.footholds(), [])

    def test_ray_cast_footholds_stay_within_their_actual_treads(self):
        for pitch in (-0.30, -0.15, 0.0, 0.15):
            profile = look(flight(), pitch=pitch)
            self.assertTrue(profile.footholds())
            for start, end, height in profile.footholds():
                level = round((height + RIDE) / RISER)
                self.assertLess(abs(height + RIDE - level * RISER), 0.015)
                left = -1 if level == 0 else 0.70 + (level - 1) * TREAD
                right = 0.70 + level * TREAD
                self.assertGreaterEqual(start, left - 0.015)
                self.assertLessEqual(end, right + 0.015)

    def test_it_finds_the_floor_under_a_standing_robot(self):
        profile = look(flight())
        self.assertTrue(profile.valid)
        self.assertAlmostEqual(profile.ground, -RIDE, delta=0.01)

    def test_flat_ground_reports_no_steps(self):
        profile = look([(-1.0, 0.0), (8.0, 0.0)])
        self.assertTrue(profile.valid)
        self.assertEqual(profile.steps, [])

    def test_a_scan_of_nothing_is_not_a_staircase(self):
        profile = perception.from_scan(-0.20, 0.0083,
                                       [float("inf")] * 200)
        self.assertFalse(profile.valid)
        self.assertIsNone(profile.next_step)

    def test_it_does_not_invent_a_step_out_of_noise(self):
        profile = look([(-1.0, 0.0), (8.0, 0.0)], noise=0.01)
        self.assertEqual(profile.steps, [])


class TestRiserHeight(unittest.TestCase):
    """The number the climb is actually driven by.

    The foot clearance is set from this, so an error here is a foot that
    catches a nosing or a leg asked for reach it does not have.
    """

    def test_it_measures_a_90_mm_riser(self):
        step = look(flight()).next_step
        self.assertIsNotNone(step)
        self.assertAlmostEqual(step.rise, RISER, delta=0.010)

    def test_it_measures_a_shallow_riser(self):
        step = look(flight(riser=0.05)).next_step
        self.assertAlmostEqual(step.rise, 0.05, delta=0.010)

    def test_it_measures_a_tall_riser(self):
        step = look(flight(riser=0.16)).next_step
        self.assertAlmostEqual(step.rise, 0.16, delta=0.015)

    def test_each_step_is_measured_on_its_own(self):
        """Not one riser height fitted to the whole flight.

        A flight with one odd step in it is the flight that trips people, and
        assuming the steps are even is how a climber walks into the odd one.
        """
        outline = [(-1.0, 0.0), (0.70, 0.0),
                   (0.70, 0.09), (1.06, 0.09),      # a 90 mm riser
                   (1.06, 0.22), (1.42, 0.22),      # then a 130 mm one
                   (5.0, 0.22)]
        profile = look(outline)
        self.assertGreaterEqual(len(profile.steps), 2)
        self.assertAlmostEqual(profile.steps[0].rise, 0.09, delta=0.012)
        self.assertAlmostEqual(profile.steps[1].rise, 0.13, delta=0.015)

    def test_a_step_too_tall_to_climb_is_not_reported_as_a_step(self):
        """A wall is not a step, however much it looks like one edge on."""
        wall = [(-1.0, 0.0), (0.90, 0.0), (0.90, 1.5), (2.0, 1.5)]
        self.assertIsNone(look(wall).next_step)


class TestEdgeDistance(unittest.TestCase):
    """Where the riser is, which decides when to stop creeping at it."""

    def test_it_places_the_edge_of_the_first_riser(self):
        step = look(flight(foot=0.70)).next_step
        self.assertAlmostEqual(step.edge, 0.70, delta=0.035)

    def test_the_edge_moves_with_the_flight(self):
        near = look(flight(foot=0.60)).next_step
        far = look(flight(foot=1.00)).next_step
        self.assertAlmostEqual(far.edge - near.edge, 0.40, delta=0.05)

    def test_reach_is_measured_from_the_front_feet_not_the_trunk(self):
        step = perception.Step(edge=0.70, rise=0.09, top=0.09)
        self.assertAlmostEqual(step.reach(), 0.70 - perception.FRONT_FOOT_X)


class TestPitch(unittest.TestCase):
    """The trunk is nose-up for the whole of a climb.

    Front feet one tread higher than the rear over a 0.387 m wheelbase is
    13 degrees, and it is the normal state of a robot on a flight rather than
    an upset. A profile read without it is wrong by about 15 mm per degree at
    half a metre, which on a 90 mm riser is not a rounding error.
    """

    NOSE_UP = -math.atan2(RISER, 0.387)

    def test_a_pitched_robot_still_measures_the_riser(self):
        step = look(flight(), pitch=self.NOSE_UP).next_step
        self.assertIsNotNone(step)
        self.assertAlmostEqual(step.rise, RISER, delta=0.012)

    def test_ignoring_the_pitch_gets_it_wrong(self):
        angle_min, increment, ranges = perception.simulate_scan(
            flight(), RIDE, pitch=self.NOSE_UP, noise=0.003)
        blind = perception.from_scan(angle_min, increment, ranges, pitch=0.0)
        self.assertTrue(blind.next_step is None
                        or abs(blind.next_step.rise - RISER) > 0.004)

    def test_a_nose_down_robot_measures_it_too(self):
        step = look(flight(), pitch=-self.NOSE_UP).next_step
        self.assertIsNotNone(step)
        self.assertAlmostEqual(step.rise, RISER, delta=0.012)


class TestDescent(unittest.TestCase):
    """Standing at the top, the next thing is a drop and not a step."""

    def test_a_drop_is_reported_with_a_negative_rise(self):
        step = look(descent()).next_step
        self.assertIsNotNone(step)
        self.assertLess(step.rise, 0.0)
        self.assertAlmostEqual(step.rise, -RISER, delta=0.012)

    def test_the_edge_of_the_drop_is_where_the_floor_ends(self):
        step = look(descent(edge=0.55)).next_step
        self.assertAlmostEqual(step.edge, 0.55, delta=0.05)


class TestScanGeometry(unittest.TestCase):
    """The conversion from beam angles to points on the ground."""

    def test_beams_above_the_horizon_are_dropped(self):
        """They are looking at walls, and a wall is not a step."""
        points = perception.scan_points(-0.20, 0.05, [1.0] * 10)
        self.assertTrue(all(forward > 0 for forward, _ in points))
        self.assertEqual(len(points), 5)

    def test_missing_returns_are_dropped(self):
        ranges = [1.0, float("inf"), float("nan"), None, 1.0]
        points = perception.scan_points(0.5, 0.1, ranges)
        self.assertEqual(len(points), 2)

    def test_returns_outside_the_range_limits_are_dropped(self):
        points = perception.scan_points(0.5, 0.1, [0.01, 1.0, 99.0],
                                        range_min=0.05, range_max=4.0)
        self.assertEqual(len(points), 1)

    def test_a_steeper_beam_lands_nearer_and_lower(self):
        near, = perception.scan_points(1.2, 0.1, [1.0])
        far, = perception.scan_points(0.4, 0.1, [1.0])
        self.assertLess(near[0], far[0])
        self.assertLess(near[1], far[1])

    def test_points_come_back_ordered_outward(self):
        points = perception.scan_points(0.2, 0.05, [2.0] * 20)
        self.assertEqual(points, sorted(points))


class TestDescription(unittest.TestCase):
    """What it says it can see, which is the only window into a live climb."""

    def test_it_says_how_far_and_how_tall(self):
        words = look(flight()).describe()
        self.assertIn("step up", words)
        self.assertIn("mm", words)

    def test_flat_ground_says_so(self):
        self.assertIn("flat", look([(-1.0, 0.0), (8.0, 0.0)]).describe())

    def test_a_drop_is_not_called_a_step_up(self):
        self.assertIn("drop", look(descent()).describe())


if __name__ == "__main__":
    unittest.main()
