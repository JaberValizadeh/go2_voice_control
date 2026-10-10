#!/usr/bin/env python3
"""Unit tests for the voice vocabulary and turn maths. No ROS master needed."""

import math
import unittest

from go2_voice_control import interpreter as it


class TestVocabulary(unittest.TestCase):

    def assert_kind(self, text, kind):
        self.assertEqual(it.interpret(text).kind, kind,
                         "%r should be %s" % (text, kind))

    def test_forward_phrases(self):
        for text in ("forward", "go forward", "move forward", "go", "ahead"):
            intent = it.interpret(text)
            self.assertEqual(intent.kind, it.WALK, text)
            self.assertEqual(intent.direction, it.FORWARD, text)

    def test_backward_phrases_walk_in_reverse(self):
        # These used to trigger a 180 degree turn, which is not what
        # "move backward" means.
        for text in ("backward", "move backward", "backwards", "reverse"):
            intent = it.interpret(text)
            self.assertEqual(intent.kind, it.WALK, text)
            self.assertEqual(intent.direction, it.BACKWARD, text)

    def test_turn_around_phrases_turn_then_drive(self):
        for text in ("turn around", "go back", "turn back", "around"):
            intent = it.interpret(text)
            self.assertEqual(intent.kind, it.TURN, text)
            self.assertAlmostEqual(intent.angle, it.HALF_TURN, msg=text)
            self.assertTrue(intent.forward_after, text)

    def test_go_back_is_not_confused_with_forward(self):
        # "go back" contains "go"; precedence must not turn it into a
        # forward walk.
        self.assertEqual(it.interpret("go back").kind, it.TURN)

    def test_quarter_turns(self):
        left = it.interpret("turn left")
        self.assertEqual(left.kind, it.TURN)
        self.assertEqual(left.direction, it.LEFT)
        self.assertAlmostEqual(left.angle, it.QUARTER_TURN)
        self.assertFalse(left.forward_after)

        right = it.interpret("turn right")
        self.assertEqual(right.direction, it.RIGHT)

    def test_stop_phrases(self):
        for text in ("stop", "halt", "freeze", "please stop"):
            self.assert_kind(text, it.STOP)

    def test_speed_phrases(self):
        for text in ("faster", "go faster", "speed up"):
            self.assertGreater(it.interpret(text).scale, 1.0, text)
        for text in ("slower", "go slower", "slow down"):
            self.assertLess(it.interpret(text).scale, 1.0, text)

    def test_speed_up_not_read_as_forward(self):
        # "go faster" and "go slower" both contain "go".
        self.assertEqual(it.interpret("go faster").kind, it.SPEED)
        self.assertEqual(it.interpret("go slower").kind, it.SPEED)

    def test_unknown_and_empty(self):
        for text in ("", None, "banana", "[unk]", "   "):
            self.assert_kind(text, it.UNKNOWN)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(it.interpret("  TURN   LEFT  ").kind, it.TURN)

    def test_unk_token_is_stripped(self):
        self.assertEqual(it.interpret("[unk] turn left [unk]").kind, it.TURN)

    def test_intents_are_immutable(self):
        # Callers must not be able to mutate a shared intent.
        with self.assertRaises(AttributeError):
            it.interpret("stop").kind = it.WALK

    def test_every_recognizer_phrase_is_understood(self):
        """The recognizer grammar and the teleop vocabulary must agree.

        A phrase Vosk is told to listen for but the teleop cannot act on is a
        silent dead end for the user.
        """
        grammar = it.RECOGNIZER_GRAMMAR
        self.assertTrue(grammar, "grammar must not be empty")
        unhandled = [p for p in grammar
                     if it.interpret(p).kind == it.UNKNOWN]
        self.assertEqual(unhandled, [])


class TestSpeedScaling(unittest.TestCase):

    def test_clamps_to_maximum(self):
        self.assertEqual(it.scale_speed(0.35, 1.5, 0.05, 0.4), 0.4)

    def test_clamps_to_minimum(self):
        self.assertEqual(it.scale_speed(0.06, 1 / 1.5, 0.05, 0.4), 0.05)

    def test_scales_within_range(self):
        self.assertAlmostEqual(it.scale_speed(0.2, 1.5, 0.05, 0.4), 0.3)

    def test_repeated_faster_then_slower_is_stable(self):
        speed = 0.2
        for _ in range(10):
            speed = it.scale_speed(speed, 1.5, 0.05, 0.4)
        self.assertEqual(speed, 0.4)
        for _ in range(20):
            speed = it.scale_speed(speed, 1 / 1.5, 0.05, 0.4)
        self.assertEqual(speed, 0.05)


class TestTurnMaths(unittest.TestCase):

    def test_normalize_wraps(self):
        for angle in (3 * math.pi, -3 * math.pi):
            # +pi and -pi are the same heading; atan2 picks the sign from the
            # sign of its zero argument, so only the magnitude is meaningful.
            self.assertAlmostEqual(abs(it.normalize_angle(angle)), math.pi)
        self.assertAlmostEqual(it.normalize_angle(0.5), 0.5)
        self.assertAlmostEqual(it.normalize_angle(2 * math.pi + 0.5), 0.5)

    def test_remaining_takes_short_way_round(self):
        # From 170 deg to -170 deg is +20 deg, not -340.
        remaining = it.turn_remaining(math.radians(-170), math.radians(170))
        self.assertAlmostEqual(remaining, math.radians(20), places=6)

    def test_finished_within_tolerance(self):
        self.assertTrue(it.turn_finished(0.01, it.LEFT, 0.05))
        self.assertFalse(it.turn_finished(0.5, it.LEFT, 0.05))

    def test_finished_on_overshoot(self):
        # Turning left but the remaining angle went negative: we passed it.
        self.assertTrue(it.turn_finished(-0.2, it.LEFT, 0.05))
        self.assertTrue(it.turn_finished(0.2, it.RIGHT, 0.05))

    def test_half_turn_start_is_not_mistaken_for_overshoot(self):
        # At the start of a 180 degree turn `remaining` sits at +/-pi and its
        # sign can flip on the first sample. That must not end the turn.
        self.assertFalse(it.turn_finished(-math.pi, it.LEFT, 0.05))
        self.assertFalse(it.turn_finished(math.pi, it.RIGHT, 0.05))

    def test_turn_speed_is_signed_and_bounded(self):
        self.assertGreater(it.turn_speed(1.0, it.LEFT, 0.4, 0.15), 0)
        self.assertLess(it.turn_speed(1.0, it.RIGHT, 0.4, 0.15), 0)
        self.assertAlmostEqual(
            abs(it.turn_speed(10.0, it.LEFT, 0.4, 0.15)), 0.4)
        self.assertAlmostEqual(
            abs(it.turn_speed(0.0001, it.LEFT, 0.4, 0.15)), 0.15)


if __name__ == "__main__":
    unittest.main()
