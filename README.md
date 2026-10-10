# Voice control

Offline voice teleoperation for a quadruped in simulation - **the Unitree Go2
by default**, or Nova-Q with `robot:=nova_q`. Vosk turns
microphone audio into a small command vocabulary; `voice_teleop.py` publishes
`geometry_msgs/Twist` for the gait controller.

Everything it needs is already installed, including the speech model at
`~/vosk-model-small-en-us-0.15`.

## Which robot this drives

**The Unitree Go2, by default.** `roslaunch go2_voice_control sim.launch`
brings up `go2_empty_world`; `robot:=nova_q` brings up Nova-Q instead.

The Go2 is the default because stairs are what this package is for, and the
Go2 is the robot that climbs them. Nova-Q walks, turns and stops by voice
perfectly well, but its stair crawl is still unsolved - see the known gap
below. Defaulting to the robot that cannot do the headline thing meant
`sim.launch` fell over on "go up" unless you knew to pass an argument.

Nothing in this package's code cares either way. The teleop, the interpreter,
the stair planner and the profile reader talk to `/cmd_vel`, `/stair_scan`,
`/stair_mode` and `/voice_commands`, none of which names a robot. Switching
is entirely a matter of which stack `sim.launch` brings up underneath, plus
the handful of numbers that describe the robot's size - and every one of
those is a conditional in that file rather than an edit.

### What differs between the two

| | Go2 | Nova-Q | because |
|---|---|---|---|
| `linear_speed` | 0.60 | **0.35** | Nova-Q's gait saturates near 0.30 m/s; see below |
| `stair_bottom_z` / `top_z` | 0.30 / 0.75 | **0.31 / 0.76** | trunk heights, and Nova-Q stands 31 mm taller |
| `stair_clearance` | 0.045 | **0.050** | a reach, and its legs are 10% longer |
| `stair_max_swing` | 0.18 | **0.20** | likewise |
| gait limb geometry | hardcoded | **parameters** | see below |

**`go2_gait.py`'s limb geometry used to be hardcoded** to the Go2's 0.213 m
segments. It is now five ROS parameters defaulting to those same values, so
the Go2 path is untouched and
`nova_q_description/launch/nova_q_walk.launch` passes Nova-Q's. Forking the
gait would have meant maintaining two copies of a 1450-line controller that
differ by five numbers.

**Teleop speed.** Commanding a gait past what it can deliver does not make a
robot quick, it makes it wander. Measured over 9 s on bare floor, commanded
against achieved, with worst yaw deviation:

| commanded | achieved | yaw wander | sideways |
|---|---|---|---|
| 0.35 | 0.24 m/s | **4°** | 0.00 m |
| 0.40 | 0.29 m/s | 12° | −0.28 m |
| 0.45 | 0.29 m/s | 10° | +0.36 m |
| 0.60 | 0.31 m/s | 26° | −0.37 m |

0.60 is the Go2's setting and it buys Nova-Q 0.07 m/s for six times the
wander. A voice-driven robot that does not go where it was pointed is worse
than a slow one.

### Known gap: Nova-Q does not climb stairs yet

This is why the Go2 is the default.

Walking, turning and stopping by voice all work on Nova-Q. **"go up" does
not.** The robot walks to the flight and goes over on the way in — measured
between x = 1.61 and x = 1.85, with the first riser at x = 2.20, so it falls
on flat floor while in the crawl gait, before reaching a step. It is
reproducible to the same spot and the same final trunk height (0.161 m).

The gait says why:

```
Stair gait blocked: gave up recovering FL: tilt 0.78 rad
Stair leg FL exceeds reach: (0.112, 0.163, -0.41)
```

The tilt is the cause and the reach error is the consequence — 0.41 m
demanded of a leg with 0.434 m of knee-limited reach, from a trunk already
at 45°. The crawl's body-shift and stance parameters are tuned for the Go2's
lighter, shorter-legged body and have not been re-derived for Nova-Q; scaling
the terrain-adaptation clamps alone did not fix it.

The Go2 climbs: verified after all of the above, trunk 0.278 m → 0.643 m and
still ascending. It is the default for that reason. Use `robot:=nova_q` for
Nova-Q on flat ground, or `nova_voice_control`, which is built for it.

## Run

Simulation and voice nodes together:

```bash
cd ~/go2_ws
source devel/setup.bash
roslaunch go2_voice_control sim.launch
```

Wait for `Listening for Go2 voice commands`, then speak.

Against an already-running simulation, start just the voice nodes:

```bash
roslaunch go2_voice_control voice_control.launch
```

## What to say

| say | the robot |
| --- | --- |
| `move forward`, `go forward`, `forward`, `go`, `ahead` | walks forward |
| `move backward`, `backward`, `reverse` | walks in reverse, keeping its heading |
| `go back`, `turn around`, `turn back` | turns 180 degrees, then walks forward |
| `turn left` / `turn right`, or just `left` / `right` | turns 90 degrees |
| `faster`, `speed up` / `slower`, `slow down` | changes speed by 1.5x |
| `go up`, `upstairs`, `climb the stairs`, `take the stairs` | walks to the stairs and tries to climb (see below) |
| `come down`, `go down`, `downstairs` | the same, downward |
| `stop`, `halt`, `freeze` | stops, including a run in progress |

`backward` and `go back` mean different things: the first reverses, the second
turns around. The bare word "back" is not in the vocabulary, because the
recognizer emits it for both.

`speed up` and `slow down` still change speed, and that ordering is
load-bearing: both contain a stair word, phrases are matched before single
words, and those two are matched before the stair phrases. So "slow down" is a
speed change and "come down" is a staircase, which is what a person means by
each.

Walking stops on its own five seconds after the last command. Pass
`command_timeout:=0` to keep going until told otherwise.

## Going up and down

```bash
roslaunch go2_voice_control sim.launch
```

The default world, `stair_room.world`, contains five solid 90 mm risers,
0.36 m treads, and a platform 0.45 m above the floor. Say `go up` to start
an ascent. Allow several minutes for the slow crawl to finish.

### Why the climb takes about two and a half minutes

Measured over 4 headless climbs, all successful: 150–157 s from "go up" to
standing on the platform. About 38 s of that is getting to the first riser,
80 s is the five risers, and 25–40 s is bringing the rear feet up onto the
platform.

On the flight, the time is foot placements: about 50 of them at 2.1 s each.
A crawl moves one leg at a time, so each placement advances the body a
quarter of the 0.24 m stride. Making the climb much faster means fewer
placements (a longer stride) or quicker ones, and both have been tried:

| change | climbs passed | time to platform when it passed |
| --- | --- | --- |
| none | 4 of 4 | 147–157 s |
| don't stop to look for each riser, full stride | 0 of 2 | – |
| don't stop to look for each riser, short stride | 2 of 3 | 142–148 s |
| quicker swing (lift 0.16 s, reach 0.22 s, lower 0.35 s) | 1 of 3 | 127 s |
| trot to the first riser at 0.30 m/s (`stair_walkup_speed:=0.30`) | 4 of 7 | 124–148 s |

A climb that fails is worse than a slow one, so none of these is on by
default. The faster walk-up is left in as an option because it doesn't
touch the crawl and its failures may be chance, but it needs more climbs to
tell.

A climb that is several times faster needs a different gait on the stairs,
one that moves two legs at a time, not a faster crawl.

### Why a climb sometimes fails

The original crawl reached the platform in 4 of 6 headless climbs. Every
failure seen, and the evidence for it:

1. **A foot planted against the first riser.** The third placement lifts
   the front-right foot. When no measured tread landing is in reach, the
   crawl lands it blind on the floor at the same stride, and nothing checks
   that point against the riser. It went down 1–2 cm short of the riser
   face, scraped it, the leg ran out of reach and the trunk rolled over.
   The scanner had the riser position to within a centimetre. (Once the
   crawl starts, the floor in front of the flight is no longer in the
   scanned treads at all.)
2. **A shift that can never earn its margin.** One leg's body shift cannot
   bring the COM 2 cm inside the other three feet. Recovery retries the same
   leg from the same triangle, and it also waits for four loaded feet while
   the body is still leaning off the foot that was about to lift. That foot
   never loads, so recovery times out and the crawl blocks.
3. **A rear leg left out of reach mid-flight.** The trunk creeps forward
   during a shift until a rear leg can no longer reach its tread.

Fixes for 1 and 2 were built and unit-tested:

- Skip a blind landing within 10 cm of a measured riser.
- Retry a failed shift on the next leg.
- Press an unloaded foot down during recovery, as the shift and the settle
  already do.

Each did what it targets in Gazebo: no blind landing reached the riser, and
one retry visibly rescued a placement. But together they passed 3 of 6
climbs, and 3 of those climbs shuffled for a minute or more near the fourth
riser, which the original never did. So they were reverted. Telling a real
improvement from noise at these success rates needs about 20 climbs per
variant, not 6. `tools/validate_stair_climb.py` makes that an unattended
overnight run.

The robot walks to the staging point, aligns with the flight, and changes to
a contact-driven crawl when the muzzle scanner measures a riser. The crawl
moves the body over three supporting feet before lifting the fourth. Planted
feet keep their world positions; the swinging foot lifts vertically, moves
into a measured tread, and lowers until its position and contact are confirmed.
The body pitch follows the difference between front and rear foothold heights.
The controller checks the measured support triangle before lifting a foot and
waits for stable contacts between steps.

Overlapping scans estimate tread heights, and landing positions leave space
for the foot and calf at risers. The gait waits when support is missing or a
landing disagrees with the measured tread. A missing or stale scanner stops
the traverse. `stop` finishes an active landing and holds the existing stance
when the feet are on different steps.

This controller uses Gazebo body pose, joint states, and foot contacts. It is
simulation locomotion, not a hardware controller for a physical Go2. Descending
solid stairs remains experimental.

The previous ramp demonstration is still available explicitly:

```bash
roslaunch go2_voice_control sim.launch use_stair_scanner:=false \
  world:=$(rospack find go2_voice_control)/worlds/stair_room_ramp.world
```

That world has a smooth supporting incline under visual steps. It does not
validate stair climbing.

### Where the flight is

Given rather than discovered, and that is a real limit: a horizontal lidar at
knee height sees a rising surface as a wall, because from there that is exactly
what it looks like.

```bash
roslaunch go2_voice_control sim.launch \
  stair_foot:="[2.20, 0.0]" stair_top:="[4.00, 0.0]" \
  stair_bottom_z:=0.30 stair_top_z:=0.75 stair_heading:=0.0
```

Those are **trunk** heights. The run checks both height and progress beyond
the flight endpoint so reaching the final riser does not count as reaching
the platform. Changes in commanded body height are accounted for separately.

The stair-run state machine is in `src/go2_voice_control/stairs.py`, and the
footstep planner is in `go2_empty_world/src/go2_empty_world/crawl.py`.

## Options

```bash
roslaunch go2_voice_control voice_control.launch \
  device:=hw:1,0 gain:=2.0 command_timeout:=10.0
```

| arg | default | meaning |
| --- | --- | --- |
| `device` | `default` | ALSA capture device; `arecord -l` lists them |
| `gain` | 1.0 | input gain, raise for a quiet microphone |
| `model_path` | `~/vosk-model-small-en-us-0.15` | unpacked Vosk model |
| `command_timeout` | 5.0 | seconds of silence before stopping; 0 disables |
| `cmd_vel_topic` | `/cmd_vel` | where Twist commands are published |
| `stair_walkup_speed` | 0.0 | trot speed to the first riser, m/s; 0 = crawl creep. Experimental, see above |

## Without a microphone

Send the same commands as text:

```bash
rostopic pub -1 /voice_commands std_msgs/String "data: 'move forward'"
rostopic pub -1 /voice_commands std_msgs/String "data: 'turn left'"
rostopic pub -1 /voice_commands std_msgs/String "data: 'stop'"
```

## If it does not respond

Check the recognizer is hearing you - it logs every phrase it recognizes:

```bash
rostopic echo /voice_commands
```

Nothing appearing means the microphone or model is the problem, not the robot.
Raise `gain`, or pick the right `device` from `arecord -l`.

## Tests

```bash
cd ~/go2_ws
catkin_make run_tests_go2_voice_control run_tests_go2_empty_world
catkin_test_results
```

Tests cover speech commands, stair sequencing, scanner processing, foot placement,
support and gait transitions, plus teleop integration over ROS topics. These
checks complement a full Gazebo climb; passing unit tests alone does not prove
that the robot can climb the flight.

Run the headless full-climb check without a microphone:

```bash
source devel/setup.bash
python3 src/go2_voice_control/tools/validate_stair_climb.py
```

It uses separate ROS/Gazebo ports, checks arrival and five seconds of standing
on the platform, and writes logs and `result.json` to `build/stair_validation`.
An occupied test port is rejected; use `--ros-port` and `--gazebo-port` to
choose other ports. A failed climb returns a nonzero exit code.

## Layout

```
src/go2_voice_control/interpreter.py   speech -> intent, pure, no rospy
src/go2_voice_control/stairs.py        the up/down errand, pure, no rospy
scripts/voice_teleop.py                intent -> Twist, turns, stair runs, timeout
scripts/voice_recognizer.py            microphone -> text, via Vosk
tools/make_room.py                     generates worlds/stair_room.world
worlds/stair_room.world                platform, steps, kerbs (generated)
test/                                  unit and integration tests
```
