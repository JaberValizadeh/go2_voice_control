#!/usr/bin/env python3
"""Climbing one riser at a time: the sequencer, without a simulator.

The flight is ray cast, so the sequencer is fed the real perception path.
The robot below is a toy, but not a forgiving one: its feet only transfer to
the next tread when the swing height it was handed actually clears the riser,
and its trunk rides at the mean of its two ends. A sequencer that gets the
clearance wrong gets stuck here, which is the point.
"""

import unittest

from go2_voice_control import perception, stepper


RISER, TREAD, COUNT = 0.09, 0.36, 5
WHEELBASE, RIDE = 0.387, 0.30


class Flight(object):
    """A toy robot on a ray-cast staircase."""

    def __init__(self, riser=RISER, tread=TREAD, count=COUNT, start=-0.70):
        self.riser, self.tread, self.count = riser, tread, count
        self.outline = perception.staircase_outline(0.0, riser, tread, count)
        self.x, self.z = start, 0.0
        self.levels = {"front": 0, "rear": 0}

    def look(self):
        local = [(px - self.x, pz - self.z) for px, pz in self.outline]
        angle_min, increment, ranges = perception.simulate_scan(
            local, RIDE, noise=0.003)
        return perception.from_scan(angle_min, increment, ranges)

    def advance(self, plan, dt):
        self.x += 0.30 * plan.speed * dt
        for name, offset in (("front", +WHEELBASE / 2.0),
                             ("rear", -WHEELBASE / 2.0)):
            foot = self.x + offset
            under = 0
            for index in range(self.count):
                if foot >= index * self.tread:
                    under = index + 1
            if under > self.levels[name]:
                if plan.swing_height >= self.riser + 0.004:
                    self.levels[name] += 1
            elif under < self.levels[name]:
                self.levels[name] = under
        wanted = self.riser * (self.levels["front"] + self.levels["rear"]) / 2.0
        self.z += (wanted - self.z) * min(1.0, dt / 0.35)

    @property
    def climbed(self):
        return self.levels["front"] >= self.count \
            and self.levels["rear"] >= self.count


def fly(sequencer, world=None, limit=120.0, dt=0.05):
    """Run a sequencer against a toy flight. Returns the states it went through."""
    world = world or Flight()
    states, swings = [], []
    now, grace = 0.0, 2.0
    while now < limit and sequencer.state != stepper.STALLED:
        plan = sequencer.update(world.z + RIDE, world.look(), now)
        if not states or states[-1] != plan.state:
            states.append(plan.state)
        swings.append(plan.swing_height)
        world.advance(plan, dt)
        now += dt
        if world.climbed:
            grace -= dt
            if grace <= 0.0:
                break
    return states, swings, world, now


class Seen(object):
    """A profile that always reports the same thing, for the timing tests."""

    valid = True

    def __init__(self, rise=RISER, edge=0.60):
        self.steps = [perception.Step(edge=edge, rise=rise, top=rise)]
        self.next_step = self.steps[0]


class Flat(object):
    valid = True
    steps = []
    next_step = None


class TestClimbing(unittest.TestCase):
    """The whole errand, up a flight it has never been told the shape of."""

    def test_it_climbs_the_whole_flight(self):
        sequencer = stepper.StepSequencer()
        _, _, world, _ = fly(sequencer)
        self.assertTrue(world.climbed)
        self.assertAlmostEqual(world.z, RISER * COUNT, delta=0.01)

    def test_it_counts_the_steps_it_climbed(self):
        sequencer = stepper.StepSequencer()
        fly(sequencer)
        self.assertEqual(sequencer.index, COUNT)

    def test_it_goes_one_at_a_time_rather_than_in_one_run(self):
        sequencer = stepper.StepSequencer()
        states, _, _, _ = fly(sequencer)
        self.assertGreaterEqual(states.count(stepper.LIFT), COUNT - 1)

    def test_it_stops_to_look_before_each_step(self):
        sequencer = stepper.StepSequencer()
        states, _, _, _ = fly(sequencer)
        self.assertGreaterEqual(states.count(stepper.LOOK), COUNT)

    def test_it_climbs_a_flight_of_a_different_pitch_without_being_retuned(self):
        sequencer = stepper.StepSequencer()
        _, swings, world, _ = fly(sequencer,
                                  Flight(riser=0.06, tread=0.30, count=6))
        self.assertTrue(world.climbed)
        self.assertAlmostEqual(max(swings), 0.06 + sequencer.clearance,
                               delta=0.02)

    def test_it_does_not_stall_on_a_flight_it_can_climb(self):
        sequencer = stepper.StepSequencer()
        fly(sequencer)
        self.assertNotEqual(sequencer.state, stepper.STALLED)


class TestSwingHeight(unittest.TestCase):
    """The foot clearance, which is the point of measuring anything.

    A blind gait lifts a fixed amount: too little catches the nosing, too much
    lurches on the flat and runs the leg out of reach.
    """

    def test_it_comes_from_the_riser_that_was_measured(self):
        sequencer = stepper.StepSequencer(clearance=0.045)
        _, swings, _, _ = fly(sequencer)
        self.assertAlmostEqual(max(swings), RISER + 0.045, delta=0.02)

    def test_a_taller_riser_gets_a_higher_lift(self):
        low = stepper.StepSequencer()
        high = stepper.StepSequencer()
        _, low_swings, _, _ = fly(low, Flight(riser=0.06, tread=0.30,
                                              count=6))
        _, high_swings, _, _ = fly(high, Flight(riser=0.12, tread=0.40,
                                                count=4))
        self.assertGreater(max(high_swings), max(low_swings) + 0.04)

    def test_it_never_asks_for_more_reach_than_the_leg_has(self):
        sequencer = stepper.StepSequencer(max_swing=0.18)
        _, swings, _, _ = fly(sequencer, Flight(riser=0.30, tread=0.50,
                                                count=2), limit=40.0)
        self.assertLessEqual(max(swings), 0.18)

    def test_a_drop_is_not_given_extra_lift(self):
        """Going down, the foot steps into space; it is the climb that needs
        the riser cleared."""
        sequencer = stepper.StepSequencer(direction=-1.0)
        now = 0.0
        while sequencer.state == stepper.LOOK and now < 5.0:
            plan = sequencer.update(0.30, Seen(rise=-RISER), now)
            now += 0.05
        self.assertAlmostEqual(plan.swing_height, sequencer.min_swing)


class TestLooking(unittest.TestCase):
    """Standing still to measure, and refusing to act on one sweep."""

    def test_it_stands_still_while_it_looks(self):
        sequencer = stepper.StepSequencer()
        plan = sequencer.update(0.30, Seen(), 0.0)
        self.assertEqual(plan.state, stepper.LOOK)
        self.assertEqual(plan.speed, 0.0)

    def test_one_sweep_is_not_enough_to_commit(self):
        sequencer = stepper.StepSequencer(look_samples=5)
        for tick in range(3):
            sequencer.update(0.30, Seen(), tick * 0.05)
        self.assertEqual(sequencer.state, stepper.LOOK)

    def test_sweeps_that_agree_commit(self):
        sequencer = stepper.StepSequencer(look_samples=5)
        for tick in range(6):
            sequencer.update(0.30, Seen(), tick * 0.05)
        self.assertEqual(sequencer.state, stepper.MOUNT)
        self.assertAlmostEqual(sequencer.riser, RISER, delta=0.001)

    def test_sweeps_that_disagree_do_not(self):
        """A bad vantage point reads differently every sweep, which is what
        makes agreement a real filter rather than a formality."""
        sequencer = stepper.StepSequencer(look_samples=5, look_spread=0.01,
                                          look_timeout=99.0)
        for tick in range(20):
            rise = 0.06 if tick % 2 else 0.13
            sequencer.update(0.30, Seen(rise=rise), tick * 0.05)
        self.assertEqual(sequencer.state, stepper.LOOK)

    def test_a_bump_too_small_to_matter_is_not_a_step(self):
        sequencer = stepper.StepSequencer(min_rise=0.03, look_timeout=99.0)
        for tick in range(20):
            sequencer.update(0.30, Seen(rise=0.01), tick * 0.05)
        self.assertEqual(sequencer.state, stepper.LOOK)


class TestSeeingNothing(unittest.TestCase):
    """The last riser of a flight cannot be seen from in front of it.

    The scanner's nearest beams land about 0.18 m ahead of the front feet, so
    by the time the robot is at the top riser it is under the near limit. That
    is the normal end of every flight, not a failure, and standing there
    waiting to see it would mean never finishing.
    """

    def test_it_does_not_stand_still_forever(self):
        sequencer = stepper.StepSequencer(look_timeout=0.5)
        now = 0.0
        while now < 2.0:
            plan = sequencer.update(0.30, Flat(), now)
            now += 0.05
        self.assertEqual(plan.state, stepper.SEARCH)
        self.assertGreater(plan.speed, 0.0)

    def test_it_creeps_on_the_last_clearance_it_measured(self):
        sequencer = stepper.StepSequencer(look_timeout=0.5)
        now = 0.0
        for _ in range(10):                 # measure a riser first
            sequencer.update(0.30, Seen(), now)
            now += 0.05
        measured = sequencer.swing_height
        while sequencer.state != stepper.SEARCH and now < 20.0:
            plan = sequencer.update(0.30, Flat(), now)
            now += 0.05
        self.assertAlmostEqual(plan.swing_height, measured, delta=1e-9)

    def test_it_never_decides_the_flight_is_over(self):
        """That is stairs.py's call, on height. This module only ever reports
        that it has climbed one more step."""
        sequencer = stepper.StepSequencer(look_timeout=0.3)
        now = 0.0
        while now < 20.0:
            sequencer.update(0.30, Flat(), now)
            now += 0.05
        self.assertIn(sequencer.state, (stepper.LOOK, stepper.SEARCH))


class TestGivingUp(unittest.TestCase):
    """A step that will not go, which has to be noticed rather than repeated."""

    def stick(self, **kwargs):
        settings = dict(lift_timeout=1.0, settle_seconds=0.1)
        settings.update(kwargs)
        sequencer = stepper.StepSequencer(**settings)
        now, tried = 0.0, set()
        while now < 60.0 and sequencer.state != stepper.STALLED:
            plan = sequencer.update(0.30, Seen(edge=0.25), now)
            if plan.state == stepper.LIFT:
                tried.add(round(plan.swing_height, 4))
            now += 0.05
        return sequencer, tried

    def test_a_step_that_will_not_go_is_abandoned(self):
        sequencer, _ = self.stick()
        self.assertEqual(sequencer.state, stepper.STALLED)
        self.assertTrue(sequencer.update(0.30, Seen(), 99.0).stalled)

    def test_but_only_after_lifting_higher_first(self):
        _, tried = self.stick(retries=2)
        self.assertGreaterEqual(len(tried), 3)

    def test_each_retry_lifts_higher_than_the_last(self):
        _, tried = self.stick(retries=2, retry_clearance=0.025)
        ordered = sorted(tried)
        self.assertAlmostEqual(ordered[1] - ordered[0], 0.025, delta=1e-6)

    def test_the_reason_says_what_it_tried(self):
        sequencer, _ = self.stick()
        self.assertIn("clearance", sequencer.note)

    def test_a_stalled_sequencer_commands_nothing(self):
        sequencer, _ = self.stick()
        self.assertEqual(sequencer.update(0.30, Seen(), 99.0).speed, 0.0)


class TestHeightAccounting(unittest.TestCase):
    """Counting risers, which is the only thing that says a step was climbed.

    A crawl lifts the trunk half a riser per foot, so a step is usually part
    done by the time the previous one is called finished. The datum has to
    carry that remainder forward or the top of a flight goes uncounted.
    """

    def climb_by(self, sequencer, heights, profile=None):
        now = 0.0
        for _ in range(10):
            sequencer.update(heights[0], profile or Seen(), now)
            now += 0.05
        for height in heights:
            sequencer.update(height, profile or Seen(edge=0.25), now)
            now += 0.05
        return sequencer

    def test_half_a_riser_is_not_a_step(self):
        sequencer = stepper.StepSequencer()
        self.climb_by(sequencer, [0.30 + 0.045 * tick / 10.0
                                  for tick in range(11)])
        self.assertEqual(sequencer.index, 0)

    def test_a_whole_riser_is(self):
        sequencer = stepper.StepSequencer()
        self.climb_by(sequencer, [0.30 + RISER * tick / 10.0
                                  for tick in range(11)])
        self.assertEqual(sequencer.index, 1)

    def test_five_risers_are_five_steps_and_not_seven(self):
        """The trigger is a fraction of a riser; the datum moves by the whole
        of one. Using the fraction for both counts a five-step flight seven
        times."""
        sequencer = stepper.StepSequencer(settle_seconds=0.0)
        now = 0.0
        for tick in range(0, 501):
            height = 0.30 + RISER * COUNT * tick / 500.0
            sequencer.update(height, Seen(edge=0.25), now)
            now += 0.02
        self.assertEqual(sequencer.index, COUNT)

    def test_going_down_counts_drops_not_climbs(self):
        sequencer = stepper.StepSequencer(direction=-1.0, settle_seconds=0.0)
        now = 0.0
        for tick in range(0, 501):
            height = 0.30 - RISER * COUNT * tick / 500.0
            sequencer.update(height, Seen(rise=-RISER, edge=0.25), now)
            now += 0.02
        self.assertEqual(sequencer.index, COUNT)


if __name__ == "__main__":
    unittest.main()
