#!/usr/bin/env python3
"""Generate the voice-control room: a platform reached by a flight of steps.

The steps are real. Each tread is a solid box with a vertical riser face, and
there is nothing hidden underneath them - no ramp, no smoothing, no supporting
incline. A foot that lands short of a nosing hits the riser, and a foot that
lands on a tread stands 90 mm above the one behind it. That is the whole point
of the world: the robot has to see each riser and step over it.

This used to be a lie. Six visible step boxes sat on one smooth 11.8 degree
collision surface, so the robot was walking up a ramp that looked like a
staircase. It worked, and it demonstrated nothing. What made the honest
version possible was not the world but the gait: a statically stable crawl
that swings one foot at a time, per-leg ground memory from the foot contacts,
and a swing height set from the riser the scanner actually measured. See
`stair_mode` in go2_empty_world/scripts/go2_gait.py.

Geometry, and why:

* 90 mm risers. Deeper than the 75 mm the ramp used, because a riser the
  scanner cannot resolve is not a step the robot can be said to have seen -
  and shallower than a building-code stair, because the leg has 0.426 m of
  reach and the trunk rides at 0.26 m of it.
* 0.36 m treads. The wheelbase is 0.387 m, so front and rear feet always
  stand on adjacent treads rather than on the same one - the trunk pitches by
  one riser over the wheelbase and stays there, which is a steady state the
  foot levelling can hold. Treads near the wheelbase would put both ends on
  the same tread half the time and on two treads the rest, which is a trunk
  that pitches back and forth every stride.
* A 1.6 m wide flight with a kerb down each side, so a robot that wanders
  cannot step off into thin air.

The platform keeps its kerb, and that is not decoration. An earlier descent
walked off the 0.45 m drop sideways rather than using the way down, and
reported success, because it had indeed ended up at the bottom. Real raised
platforms have edge protection for exactly this reason; the only opening is at
the head of the stairs.

    ./make_room.py                    # writes worlds/stair_room.world
    ./make_room.py --riser 0.12       # a harder flight
    ./make_room.py --ramp             # writes worlds/stair_room_ramp.world

`--ramp` regenerates the world this replaced: the same visible steps, with no
collision of their own, sitting on one smooth inclined surface. It is kept
because the open-loop gait can traverse it end to end and the discrete flight
it cannot, so it is the version to launch to see the whole errand run through.
It is an accommodation and it is labelled as one - a robot walking up it is
walking up a ramp that looks like a staircase, and the scanner reports it
correctly as a smooth slope with no steps in it.
"""

import argparse
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)

RISE = 0.45
RISER, TREAD, STAIR_W = 0.09, 0.36, 1.6
PLAT = [4.0, 8.5, -3.0, 3.0]          # xmin xmax ymin ymax
STAIR_Y = 0.0                          # centred: the only way up
KERB_H, KERB_T = 0.16, 0.12

CONCRETE = (0.58, 0.57, 0.55)
STEP = (0.70, 0.68, 0.63)
NOSING = (0.86, 0.78, 0.42)
KERB = (0.72, 0.62, 0.28)


def box(name, x, y, z, sx, sy, sz, rgb, mu=1.2):
    return """    <model name="%s">
      <static>true</static>
      <pose>%.4f %.4f %.4f 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>%.4f %.4f %.4f</size></box></geometry>
          <surface><friction><ode><mu>%.2f</mu><mu2>%.2f</mu2></ode></friction>
            <contact><ode><kp>1000000</kp><kd>100</kd></ode></contact></surface></collision>
        <visual name="v"><geometry><box><size>%.4f %.4f %.4f</size></box></geometry>
          <material><ambient>%.2f %.2f %.2f 1</ambient><diffuse>%.2f %.2f %.2f 1</diffuse></material></visual>
      </link>
    </model>
""" % (name, x, y, z, sx, sy, sz, mu, mu, sx, sy, sz,
       rgb[0] * 0.45, rgb[1] * 0.45, rgb[2] * 0.45, rgb[0], rgb[1], rgb[2])


def stair_support(x0, run, rise, y, width):
    """Thin collision box whose top face joins the floor to the platform.

    Only used by --ramp. The offset puts the centre of the box's top face
    exactly on the floor-to-platform line rather than the box's own centre.
    """
    angle = math.atan2(rise, run)
    length = math.hypot(run, rise)
    thickness = 0.04
    x = x0 + run / 2.0 + math.sin(angle) * thickness / 2.0
    z = rise / 2.0 - math.cos(angle) * thickness / 2.0
    return """    <model name="stair_support">
      <static>true</static>
      <pose>%.4f %.4f %.4f 0 %.6f 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>%.4f %.4f %.4f</size></box></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
      </link>
    </model>
""" % (x, y, z, -angle, length, width, thickness)


def visual_box(name, x, y, z, sx, sy, sz, rgb):
    """Paint only - no collision. Used for the nosing stripes."""
    return """    <model name="%s">
      <static>true</static>
      <pose>%.4f %.4f %.4f 0 0 0</pose>
      <link name="link">
        <visual name="v"><geometry><box><size>%.4f %.4f %.4f</size></box></geometry>
          <material><ambient>%.2f %.2f %.2f 1</ambient><diffuse>%.2f %.2f %.2f 1</diffuse>
            <emissive>%.2f %.2f %.2f 1</emissive></material></visual>
      </link>
    </model>
""" % (name, x, y, z, sx, sy, sz,
       rgb[0] * 0.45, rgb[1] * 0.45, rgb[2] * 0.45, rgb[0], rgb[1], rgb[2],
       rgb[0] * 0.25, rgb[1] * 0.25, rgb[2] * 0.25)


HEAD = """<?xml version="1.0"?>
<!-- Generated by tools/make_room.py - do not edit by hand. -->
<sdf version="1.6">
  <world name="stair_room">
    <scene><ambient>0.62 0.62 0.64 1</ambient>
      <background>0.66 0.72 0.82 1</background><shadows>true</shadows></scene>
    <light name="sun" type="directional"><cast_shadows>true</cast_shadows>
      <pose>0 0 14 0 0 0</pose><diffuse>0.90 0.88 0.83 1</diffuse>
      <specular>0.18 0.18 0.17 1</specular>
      <direction>-0.42 0.34 -0.84</direction></light>
    <light name="fill" type="directional"><cast_shadows>false</cast_shadows>
      <pose>0 0 16 0 0 0</pose><diffuse>0.34 0.36 0.42 1</diffuse>
      <specular>0.02 0.02 0.03 1</specular>
      <direction>0.48 -0.40 -0.78</direction></light>
    <physics type="ode"><max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate></physics>
    <model name="floor"><static>true</static><link name="l">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
        <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>
        <material><ambient>0.24 0.25 0.24 1</ambient><diffuse>0.46 0.48 0.46 1</diffuse></material></visual>
    </link></model>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out")
    parser.add_argument("--ramp", action="store_true",
                        help="visual steps on a smooth incline, which this "
                             "gait can actually traverse end to end")
    parser.add_argument("--riser", type=float, default=RISER,
                        help="riser height in metres (default %.3f)" % RISER)
    parser.add_argument("--tread", type=float, default=TREAD,
                        help="tread depth in metres (default %.2f)" % TREAD)
    args = parser.parse_args()
    out = args.out or os.path.join(
        PACKAGE, "worlds",
        "stair_room_ramp.world" if args.ramp else "stair_room.world")

    riser, tread = args.riser, args.tread
    steps = int(round(RISE / riser))
    riser = RISE / steps               # divide the rise exactly
    stair_run = steps * tread
    stair_foot = PLAT[0] - stair_run

    parts = [HEAD]
    parts.append(box("platform", (PLAT[0] + PLAT[1]) / 2.0,
                     (PLAT[2] + PLAT[3]) / 2.0, RISE / 2.0,
                     PLAT[1] - PLAT[0], PLAT[3] - PLAT[2], RISE, CONCRETE))

    # Each step is one solid block from the floor up to its own tread, butted
    # against the one behind it. Its front face IS the riser - there is
    # nothing under the flight to walk up instead.
    shape = visual_box if args.ramp else box
    for i in range(steps):
        top = riser * (i + 1)
        parts.append(shape(
            "step_%d" % i, stair_foot + tread * (i + 0.5),
            STAIR_Y, top / 2.0, tread, STAIR_W, top, STEP))
        # A painted nosing on the lip of each tread. It is 2 mm proud and has
        # no collision, so it changes nothing the robot walks on; it is there
        # because a flight of same-coloured concrete blocks is genuinely hard
        # to count in a screenshot, and every stair photograph in this README
        # is somebody trying to see which step the robot is on.
        parts.append(visual_box(
            "step_nosing_%d" % i, stair_foot + tread * i + 0.03,
            STAIR_Y, top + 0.001, 0.06, STAIR_W, 0.004, NOSING))

    if args.ramp:
        parts.append(stair_support(stair_foot, stair_run, RISE, STAIR_Y,
                                   STAIR_W))

    # Kerbs down both sides of the flight, rising with it. The platform kerb
    # alone was not enough: the robot went through the opening and then
    # stepped off the SIDE of the flight, dropping 0.39 m. A flight with a
    # drop either side has edge protection in reality for the same reason.
    for side in (-1, 1):
        for i in range(steps):
            top = riser * (i + 1)
            parts.append(box(
                "stair_kerb_%s_%d" % ("s" if side < 0 else "n", i),
                stair_foot + tread * (i + 0.5),
                STAIR_Y + side * (STAIR_W / 2.0 + KERB_T / 2.0),
                (top + KERB_H) / 2.0,
                tread, KERB_T, top + KERB_H, KERB))

    # Kerb round the platform, with a gap at the head of the stairs. Segments
    # are given as (name, from_y, to_y) along the leading edge, and as whole
    # runs along the other three sides.
    gaps = [(STAIR_Y - STAIR_W / 2.0, STAIR_Y + STAIR_W / 2.0)]
    edges = []
    cursor = PLAT[2]
    for low, high in gaps:
        if low > cursor:
            edges.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < PLAT[3]:
        edges.append((cursor, PLAT[3]))
    for index, (low, high) in enumerate(edges):
        parts.append(box("kerb_front_%d" % index, PLAT[0] + KERB_T / 2.0,
                         (low + high) / 2.0, RISE + KERB_H / 2.0,
                         KERB_T, high - low, KERB_H, KERB))
    for name, x, y, sx, sy in (
            ("back", PLAT[1] - KERB_T / 2.0, (PLAT[2] + PLAT[3]) / 2.0,
             KERB_T, PLAT[3] - PLAT[2]),
            ("left", (PLAT[0] + PLAT[1]) / 2.0, PLAT[3] - KERB_T / 2.0,
             PLAT[1] - PLAT[0], KERB_T),
            ("right", (PLAT[0] + PLAT[1]) / 2.0, PLAT[2] + KERB_T / 2.0,
             PLAT[1] - PLAT[0], KERB_T)):
        parts.append(box("kerb_%s" % name, x, y, RISE + KERB_H / 2.0,
                         sx, sy, KERB_H, KERB))

    for name, x, y, sx, sy in (("n", 0, 6.0, 24.0, 0.3),
                               ("s", 0, -6.0, 24.0, 0.3),
                               ("w", -6.0, 0, 0.3, 12.0),
                               ("e", 11.0, 0, 0.3, 12.0)):
        parts.append(box("wall_%s" % name, x, y, 0.75, sx, sy, 1.5, CONCRETE))

    parts.append("""    <gui><camera name="user_camera">
      <pose>-6 -8 5 0 0.35 0.9</pose></camera></gui>
  </world>
</sdf>
""")
    with open(out, "w") as handle:
        handle.write("".join(parts))

    print("world -> %s" % out)
    print("steps: %d x %.4f m riser, %.2f tread, %.1f m wide, y=%.1f, "
          "x %.2f -> %.2f" % (steps, riser, tread, STAIR_W, STAIR_Y,
                              stair_foot, PLAT[0]))
    if args.ramp:
        print("RAMP: the steps are paint. What the robot walks on is one "
              "smooth %.1f degree surface underneath them."
              % math.degrees(math.atan2(RISE, stair_run)))
    else:
        print("every riser is a real vertical face; there is nothing under "
              "the flight to walk up instead")
    print("kerb : %.2f m high, %d segment(s) on the platform edge (open at"
          " the stairs), plus both sides of the flight"
          % (KERB_H, len(edges)))
    print()
    print("flight parameters for the teleop (TRUNK heights; it stands ~0.30):")
    print("  stair_foot: [%.2f, %.1f]" % (stair_foot, STAIR_Y))
    print("  stair_top:  [%.2f, %.1f]" % (PLAT[0], STAIR_Y))
    print("  stair_bottom_z: 0.30   stair_top_z: %.2f" % (RISE + 0.30))
    print("  stair_riser:    %.3f   stair_tread:  %.2f" % (riser, tread))


if __name__ == "__main__":
    main()
