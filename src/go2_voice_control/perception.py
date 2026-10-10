#!/usr/bin/env python3
"""Seeing a staircase: one vertical scan line in, the height of each step out.

The sensor is a single fan of ranges swept down the vertical plane straight
ahead of the muzzle - see `xacro/go2_stair_sensors.xacro` for why the plane is
vertical and not horizontal. This module is what turns that fan into the two
numbers the climb actually needs:

    how far ahead is the next riser, and how tall is it

WHAT A STAIRCASE LOOKS LIKE TO THIS SENSOR

Walk the beams outward from the steepest. They march along the tread the robot
is standing on, each landing a little further forward at the same height. At
the nosing they stop moving forward: the next several beams all strike the
vertical riser face at the same distance, climbing it. Then they clear the top
and march forward again along the next tread, one riser higher.

So the profile alternates flat runs and vertical runs, and telling them apart
is the whole algorithm: a point is on a riser when the surface through it is
steeper than 45 degrees, and on a tread when it is shallower. That test is
scale-free - it needs no threshold in metres, so it does not have to be
re-tuned for a different flight - and it is measured across a window of beams
rather than between neighbours, so single-beam noise cannot flip it.

Runs shorter than `min_run` beams are discarded rather than believed. A stray
point classified the wrong way is common; five in a row are not.

WHAT IT DELIBERATELY DOES NOT DO

It does not know that stairs are evenly spaced, and it does not fit one riser
height to the whole flight. Each step is measured on its own, immediately
before it is climbed, from a metre or less away. A flight with one odd step in
it - which is the flight that trips people, and robots - is measured correctly
step by step, and a flight that is not a flight at all reports whatever is
really there.

Deliberately free of ROS, so it can be exercised against a ray-cast staircase
with no simulator at all:

    python3 -m go2_voice_control.perception
"""

import math

# Where the scanner sits on the trunk, from go2_stair_sensors.xacro. Forward
# and up from the trunk origin, in metres.
MOUNT = (0.32715, 0.04297)

# Roughly under the front hips: the point the robot has to get a foot onto.
FRONT_FOOT_X = 0.1934


class Step(object):
    """One step seen ahead, measured rather than assumed.

    `edge` is the forward distance from the trunk origin to the riser face,
    `rise` is how much taller the tread beyond it is than the one before it
    (negative going down a flight), and `top` is that tread's height above the
    surface the robot is standing on right now.
    """

    def __init__(self, edge, rise, top):
        self.edge = edge
        self.rise = rise
        self.top = top

    def reach(self, foot_x=FRONT_FOOT_X):
        """How far the front feet are from the riser face."""
        return self.edge - foot_x

    def __repr__(self):
        return "Step(edge=%.3f, rise=%+.3f, top=%+.3f)" % (
            self.edge, self.rise, self.top)


class Profile(object):
    """What the scanner made of one sweep."""

    def __init__(self, ground, steps, treads=(), points=()):
        # Height of the surface under the robot, relative to the trunk origin.
        # Negative: the ground is below the robot.
        self.ground = ground
        self.steps = list(steps)
        self.treads = list(treads)
        self.points = list(points)

    def footholds(self, tolerance=0.012, min_span=0.15):
        """Flat scan spans suitable for a foot, excluding merged riser rays.

        The riser classifier absorbs short runs to suppress noise. At grazing
        angles that can attach a few higher rays to the end of a lower tread.
        Keep only the extent actually measured at the tread's median height.
        """
        out = []
        for start, end, z in self.treads:
            inliers = [x for x, height in self.points
                       if start <= x <= end and abs(height - z) <= tolerance]
            if inliers and max(inliers) - min(inliers) >= min_span:
                out.append((min(inliers), max(inliers), z))
        return out

    @property
    def valid(self):
        return self.ground is not None

    @property
    def next_step(self):
        return self.steps[0] if self.steps else None

    def rise_ahead(self):
        """Height of the next riser, or 0.0 when the way ahead is flat."""
        step = self.next_step
        return step.rise if step is not None else 0.0

    def describe(self):
        if not self.valid:
            return "no ground in view"
        if not self.steps:
            return "flat for as far as it can see"
        return " then ".join(
            "%.2f m ahead a %s of %.0f mm"
            % (step.reach(), "step up" if step.rise > 0 else "drop",
               abs(step.rise) * 1000.0)
            for step in self.steps[:3])


def scan_points(angle_min, angle_increment, ranges, pitch=0.0, mount=MOUNT,
                range_min=0.05, range_max=4.0, horizon=0.02):
    """Beam ranges to (forward, height) points in a gravity-aligned frame.

    The sensor link is rolled so that a beam at scan angle `a` points along
    (cos a, 0, -sin a) in the trunk frame - `a` is a depression angle below
    the trunk's own nose. Rotating that into a level frame is the one place
    the trunk's attitude enters:

        forward = r cos(a + pitch)      height = -r sin(a + pitch)

    Pitch simply adds to the beam angle, because both are rotations about the
    same axis. That is worth knowing, because it is the reason a robot part
    way up a flight - nose-up by a riser over its wheelbase - does not read
    the step in front of it as taller than it is. Without the term it would,
    by about 15 mm per degree at half a metre, which on a 90 mm riser is not
    a rounding error.

    `pitch` is positive nose-down, matching the value the gait extracts from
    the trunk quaternion.

    Beams above the horizon are dropped: they are looking at walls, and a wall
    is not a step. So are beams with no return.

    Returned points are ordered outward, nearest first.
    """
    origin_x = mount[0] * math.cos(pitch) + mount[1] * math.sin(pitch)
    origin_z = -mount[0] * math.sin(pitch) + mount[1] * math.cos(pitch)

    points = []
    for index, distance in enumerate(ranges):
        if distance is None:
            continue
        try:
            if math.isnan(distance) or math.isinf(distance):
                continue
        except TypeError:
            continue
        if distance < range_min or distance > range_max:
            continue
        depression = angle_min + index * angle_increment + pitch
        if depression <= horizon:
            continue
        points.append((origin_x + distance * math.cos(depression),
                       origin_z - distance * math.sin(depression)))
    points.sort()
    return points


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _classify(points, window):
    """True where the surface through a point is steeper than 45 degrees.

    Measured across `window` beams either side rather than between
    neighbours. Neighbouring beams are a few millimetres apart on the ground,
    which is the same size as the sensor noise, so a neighbour-to-neighbour
    slope is mostly noise. A window of five is centimetres wide on a tread and
    still nearly zero wide on a riser, which is exactly the contrast the test
    is looking for.
    """
    steep = []
    count = len(points)
    for index in range(count):
        low = max(0, index - window)
        high = min(count - 1, index + window)
        run = points[high][0] - points[low][0]
        climb = points[high][1] - points[low][1]
        steep.append(abs(climb) > abs(run))
    return steep


def _runs(flags, min_run):
    """Maximal runs of equal flags, as (value, start, stop) with stop exclusive.

    Runs shorter than `min_run` are absorbed into their neighbour rather than
    reported: a lone misclassified beam is noise, not a 4 mm step.
    """
    if not flags:
        return []
    spans = []
    start = 0
    for index in range(1, len(flags) + 1):
        if index == len(flags) or flags[index] != flags[start]:
            spans.append([flags[start], start, index])
            start = index
    merged = [spans[0]]
    for span in spans[1:]:
        if span[2] - span[1] < min_run:
            merged[-1][2] = span[2]          # too short to believe; absorb it
        elif span[0] == merged[-1][0]:
            merged[-1][2] = span[2]
        else:
            merged.append(span)
    return [(value, start, stop) for value, start, stop in merged]


def read(points, min_rise=0.025, window=3, min_run=5, look_ahead=1.6,
         max_step=0.35):
    """Segment a scan into treads and the risers between them.

    `min_rise` is the smallest height change worth calling a step; below it
    the flight is just uneven ground. `max_step` rejects a "step" taller than
    the robot could ever climb - which is what a wall, or the underside of the
    platform seen edge-on, looks like to a segmenter that trusts everything.
    """
    ahead = [point for point in points if point[0] <= look_ahead]
    if len(ahead) < 2 * min_run:
        return Profile(None, [])

    steep = _classify(ahead, window)
    treads = []
    for is_steep, start, stop in _runs(steep, min_run):
        if is_steep:
            continue
        heights = [ahead[index][1] for index in range(start, stop)]
        treads.append((ahead[start][0], ahead[stop - 1][0], _median(heights)))

    if not treads:
        return Profile(None, [])

    ground = treads[0][2]
    steps = []
    for before, after in zip(treads, treads[1:]):
        rise = after[2] - before[2]
        if abs(rise) < min_rise or abs(rise) > max_step:
            continue
        # The edge is where the tread the robot is on runs out - the last
        # thing the beams saw at that level, not the first thing they saw at
        # the next one. Those differ by the depth of the nosing, and stopping
        # a stride's length too late is how a foot ends up against a riser
        # instead of on top of it.
        steps.append(Step(edge=before[1], rise=rise, top=after[2] - ground))
    return Profile(ground, steps, treads, ahead)


def from_scan(angle_min, angle_increment, ranges, pitch=0.0, **kwargs):
    """The whole path from a LaserScan's fields to a Profile."""
    read_keys = ("min_rise", "window", "min_run", "look_ahead", "max_step")
    read_args = {key: kwargs.pop(key) for key in read_keys if key in kwargs}
    points = scan_points(angle_min, angle_increment, ranges, pitch, **kwargs)
    return read(points, **read_args)


# --------------------------------------------------------------- ray casting
#
# Used by the self-check below and by the tests. A staircase is a polyline, so
# casting a ray at one is a handful of segment intersections - which is worth
# having, because it means the segmenter can be held against a flight whose
# riser height is known exactly, with no simulator in the loop.


def staircase_outline(foot, riser, tread, count, run_out=6.0):
    """The side view of a flight, as a polyline from under the robot outward."""
    outline = [(-1.0, 0.0), (foot, 0.0)]
    for index in range(count):
        top = riser * (index + 1)
        outline.append((foot + tread * index, top))          # up the riser
        outline.append((foot + tread * (index + 1), top))    # along the tread
    outline.append((outline[-1][0] + run_out, outline[-1][1]))
    return outline


def cast(outline, origin, angle, range_max=4.0):
    """Range along a beam leaving `origin` at depression `angle`, or None."""
    direction = (math.cos(angle), -math.sin(angle))
    best = None
    for (x0, z0), (x1, z1) in zip(outline, outline[1:]):
        ex, ez = x1 - x0, z1 - z0
        denominator = direction[0] * ez - direction[1] * ex
        if abs(denominator) < 1e-12:
            continue
        px, pz = x0 - origin[0], z0 - origin[1]
        distance = (px * ez - pz * ex) / denominator
        along = (px * direction[1] - pz * direction[0]) / denominator
        if distance <= 0.0 or not -1e-9 <= along <= 1.0 + 1e-9:
            continue
        if best is None or distance < best:
            best = distance
    if best is None or best > range_max:
        return float("inf")
    return best


def simulate_scan(outline, height, angle_min=-0.20, angle_max=1.45,
                  samples=200, mount=MOUNT, pitch=0.0, noise=0.0, seed=7,
                  range_max=4.0):
    """Ranges a scanner `height` above the outline's datum would report.

    `height` is the trunk origin's height above the ground it stands on, so
    the sensor sits at `height` plus its own mounting offset.
    """
    import random
    generator = random.Random(seed)
    origin = (mount[0] * math.cos(pitch) + mount[1] * math.sin(pitch),
              height - mount[0] * math.sin(pitch) + mount[1] * math.cos(pitch))
    increment = (angle_max - angle_min) / (samples - 1)
    ranges = []
    for index in range(samples):
        angle = angle_min + index * increment + pitch
        distance = cast(outline, origin, angle, range_max)
        if noise and distance != float("inf"):
            distance += generator.gauss(0.0, noise)
        ranges.append(distance)
    return angle_min, increment, ranges


def _main():
    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        failures += 0 if ok else 1
        print("%-4s %-52s %s" % ("ok" if ok else "FAIL", label, detail))

    riser, tread = 0.09, 0.36
    outline = staircase_outline(foot=0.70, riser=riser, tread=tread, count=5)

    # Standing on the floor, a flight starting 0.70 m ahead of the trunk.
    angle_min, increment, ranges = simulate_scan(outline, 0.30, noise=0.003)
    profile = from_scan(angle_min, increment, ranges)
    check("finds the ground it is standing on",
          profile.valid and abs(profile.ground + 0.30) < 0.01,
          "z = %.3f" % (profile.ground or 0.0))
    step = profile.next_step
    check("sees a step ahead", step is not None)
    if step is not None:
        check("measures the riser to within 10 mm",
              abs(step.rise - riser) < 0.010, "%.3f m" % step.rise)
        check("places the edge to within 30 mm",
              abs(step.edge - 0.70) < 0.030, "%.3f m" % step.edge)
        check("counts more than one step ahead", len(profile.steps) >= 2,
              "%d seen" % len(profile.steps))

    # Nose-up by one riser over the wheelbase, which is how the robot stands
    # once it is on the flight. The pitch term has to earn its keep here.
    pitch = -math.atan2(riser, 0.387)
    angle_min, increment, ranges = simulate_scan(
        outline, 0.30, pitch=pitch, noise=0.003)
    tilted = from_scan(angle_min, increment, ranges, pitch=pitch)
    check("still measures the riser when nose-up %.1f deg"
          % math.degrees(-pitch),
          tilted.next_step is not None
          and abs(tilted.next_step.rise - riser) < 0.012,
          "%.3f m" % (tilted.next_step.rise if tilted.next_step else 0.0))

    uncorrected = from_scan(angle_min, increment, ranges, pitch=0.0)
    check("and ignoring the pitch would get it wrong",
          uncorrected.next_step is None
          or abs(uncorrected.next_step.rise - riser) > 0.004,
          "%.3f m" % (uncorrected.next_step.rise
                      if uncorrected.next_step else 0.0))

    # Flat floor: nothing to climb, and it must not invent anything.
    angle_min, increment, ranges = simulate_scan(
        [(-1.0, 0.0), (8.0, 0.0)], 0.30, noise=0.003)
    flat = from_scan(angle_min, increment, ranges)
    check("reports flat ground as flat", flat.valid and not flat.steps,
          flat.describe())

    # Standing at the top looking down: the next thing is a drop, not a step.
    down_outline = [(-1.0, 0.0), (0.55, 0.0)]
    for index in range(5):
        top = -riser * (index + 1)
        down_outline.append((0.55 + tread * index, top))
        down_outline.append((0.55 + tread * (index + 1), top))
    down_outline.append((down_outline[-1][0] + 6.0, down_outline[-1][1]))
    angle_min, increment, ranges = simulate_scan(down_outline, 0.30,
                                                 noise=0.003)
    descent = from_scan(angle_min, increment, ranges)
    check("reads a descent as a drop of the right size",
          descent.next_step is not None
          and abs(descent.next_step.rise + riser) < 0.012,
          "%.3f m" % (descent.next_step.rise if descent.next_step else 0.0))

    print("\n     %s" % from_scan(*simulate_scan(outline, 0.30,
                                                noise=0.003)).describe())
    print("\n%s" % ("all checks passed" if not failures
                    else "%d CHECKS FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
