"""Turn recognized speech into an intent, with no ROS dependency.

Kept free of rospy so the vocabulary can be tested directly, which matters
because speech is the one input we cannot easily produce on demand: the
recognizer needs a microphone and a downloaded model, but the decision of what
"go back" means is pure text handling and deserves to be checked in isolation.
"""

import collections
import math


# Intent kinds.
STOP = "stop"
WALK = "walk"
TURN = "turn"
SPEED = "speed"
UNKNOWN = "unknown"

QUARTER_TURN = 1.5707963267948966   # pi / 2
HALF_TURN = 3.141592653589793       # pi

LEFT = 1.0
RIGHT = -1.0

FORWARD = 1.0
BACKWARD = -1.0

SPEED_UP = 1.5
SPEED_DOWN = 1.0 / 1.5

STAIRS = "stairs"
UP = 1.0
DOWN = -1.0


Intent = collections.namedtuple(
    "Intent", "kind direction angle forward_after scale text")

# The phrase list handed to Vosk. It lives here, next to the code that acts on
# it, so the recognizer cannot be told to listen for something the teleop
# silently ignores; test_interpreter.py asserts every entry maps to an intent.
RECOGNIZER_GRAMMAR = (
    "move forward", "go forward", "forward", "forwards", "go", "ahead",
    "move backward", "backward", "backwards", "reverse",
    "go back", "turn back", "turn around", "around",
    "turn left", "left", "turn right", "right",
    "stop", "halt", "freeze",
    "faster", "go faster", "speed up",
    "slower", "go slower", "slow down",
    "go up", "up", "upstairs", "go upstairs", "climb", "climb up",
    "climb the stairs", "stairs", "take the stairs",
    "come down", "down", "go down", "downstairs", "go downstairs",
    "climb down", "come back down",
)



def _intent(kind, direction=0.0, angle=0.0, forward_after=False, scale=1.0,
            text=""):
    return Intent(kind, direction, angle, forward_after, scale, text)


# Multi-word phrases are matched before single words, because several of them
# contain a word that would otherwise trigger a different, wrong intent:
# "go back" contains "back", "speed up" contains "up", "slow down" contains
# "down". Order in this table is the precedence.
_PHRASES = (
    # "speed up" and "slow down" stay first, and that ordering is load-bearing
    # now that "up" and "down" mean stairs. Both contain a stair word, phrases
    # are tested before single words, and these two are tested before the
    # stair phrases - so "slow down" is a speed change and "come down" is a
    # staircase, which is what a person means by each of them.
    ("speed up", lambda: _intent(SPEED, scale=SPEED_UP)),
    ("slow down", lambda: _intent(SPEED, scale=SPEED_DOWN)),

    ("go upstairs", lambda: _intent(STAIRS, UP)),
    ("upstairs", lambda: _intent(STAIRS, UP)),
    ("climb the stairs", lambda: _intent(STAIRS, UP)),
    ("take the stairs", lambda: _intent(STAIRS, UP)),
    ("climb up", lambda: _intent(STAIRS, UP)),
    ("go up", lambda: _intent(STAIRS, UP)),

    ("come back down", lambda: _intent(STAIRS, DOWN)),
    ("go downstairs", lambda: _intent(STAIRS, DOWN)),
    ("downstairs", lambda: _intent(STAIRS, DOWN)),
    ("climb down", lambda: _intent(STAIRS, DOWN)),
    ("come down", lambda: _intent(STAIRS, DOWN)),
    ("go down", lambda: _intent(STAIRS, DOWN)),
    ("turn around", lambda: _intent(TURN, LEFT, HALF_TURN, True)),
    ("go back", lambda: _intent(TURN, LEFT, HALF_TURN, True)),
    ("turn back", lambda: _intent(TURN, LEFT, HALF_TURN, True)),
)

# Single words, checked after the phrases. Each maps to a factory so the
# intents stay immutable and cannot be mutated by a caller.
_WORDS = (
    (("faster", "fast"), lambda: _intent(SPEED, scale=SPEED_UP)),
    (("slower", "slow"), lambda: _intent(SPEED, scale=SPEED_DOWN)),
    (("stop", "halt", "freeze"), lambda: _intent(STOP)),
    (("around",), lambda: _intent(TURN, LEFT, HALF_TURN, True)),
    (("backward", "backwards", "reverse"),
     lambda: _intent(WALK, BACKWARD)),
    (("left",), lambda: _intent(TURN, LEFT, QUARTER_TURN)),
    (("right",), lambda: _intent(TURN, RIGHT, QUARTER_TURN)),
    (("upstairs", "up", "climb", "stairs"), lambda: _intent(STAIRS, UP)),
    (("downstairs", "down"), lambda: _intent(STAIRS, DOWN)),
    (("forward", "forwards", "ahead", "go"), lambda: _intent(WALK, FORWARD)),
)


def interpret(text):
    """Map an utterance to an Intent. Never raises; unknown input is UNKNOWN.

    "back" on its own is deliberately absent from the vocabulary. The
    recognizer emits it for both "go back" (turn around) and "move backward"
    (walk in reverse), which mean opposite things, so acting on the bare word
    would be a coin flip. The phrases that contain it are matched explicitly
    above.
    """
    if not text:
        return _intent(UNKNOWN, text="")

    cleaned = " ".join(str(text).lower().replace("[unk]", " ").split())
    if not cleaned:
        return _intent(UNKNOWN, text="")

    for phrase, factory in _PHRASES:
        if phrase in cleaned:
            return factory()._replace(text=cleaned)

    words = set(cleaned.split())
    for keywords, factory in _WORDS:
        if words.intersection(keywords):
            return factory()._replace(text=cleaned)

    return _intent(UNKNOWN, text=cleaned)


def scale_speed(current, scale, minimum, maximum):
    """Apply a speed change, clamped to [minimum, maximum]."""
    return max(minimum, min(maximum, current * scale))


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi].

    Either sign may come back at exactly half a turn; they are the same
    heading and callers must not depend on which one appears.
    """
    return math.atan2(math.sin(angle), math.cos(angle))


def turn_remaining(target_yaw, yaw):
    """Signed angle still to be turned, wrapped to [-pi, pi]."""
    return normalize_angle(target_yaw - yaw)


def turn_finished(remaining, direction, tolerance):
    """True when a turn has reached its target or gone past it.

    Overshoot is only trusted near the target. Wrapping makes the sign of
    `remaining` meaningless at the start of a half turn, where it sits at
    exactly +/-pi and can flip on the first sample.
    """
    if abs(remaining) <= tolerance:
        return True
    return remaining * direction < 0.0 and abs(remaining) < 1.0


def turn_speed(remaining, direction, max_speed, min_speed, gain=1.5):
    """Proportional turn rate, signed by direction."""
    magnitude = max(min_speed, min(max_speed, gain * abs(remaining)))
    return direction * magnitude
