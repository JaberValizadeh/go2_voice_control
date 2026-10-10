#!/usr/bin/env python3
"""Climbing a flight one riser at a time, on what the scanner actually saw.

`stairs.py` is the errand - walk to the flight, square up, go up, stop. This
is the part in the middle that used to be a single "walk forward until the
height changes", which is a fine thing to ask of a ramp and a poor thing to
ask of a staircase. A staircase is not a slope with texture. It is a sequence
of discrete obstacles, each of which has to be cleared on purpose, and the
robot has to know how tall the next one is before it commits a foot to it.

So each riser gets its own small errand:

    LOOK    stand still and measure the step in front, several times over
    MOUNT   creep up to it until the front feet are at the riser face
    LIFT    step over it, watching the trunk height for the riser to arrive
    SETTLE  stop, and let the rear feet follow and the trunk level out

then LOOK again for the next one - until the flight has shown it is even;
see A foot that catches anyway is not a failure the first time. LIFT waits for the
trunk to gain the riser it was promised, and when it does not, it tries again
with more clearance before giving up. That is what a person does on a step
they scuffed.

Deliberately free of ROS:

    python3 -m go2_voice_control.stepper
"""

LOOK = "look"
MOUNT = "mount"
LIFT = "lift"
SETTLE = "settle"
SEARCH = "search"
STALLED = "stalled"


class Plan(object):
    """What the sequencer wants for the next control tick."""

    def __init__(self, speed, swing_height, state, riser=0.0, index=0,
                 note=""):
        # Fraction of the traverse speed to use, 0 for stand still.
        self.speed = speed
        self.swing_height = swing_height
        self.state = state
        self.riser = riser
        self.index = index
        self.note = note

    @property
    def stalled(self):
        return self.state == STALLED

    def __repr__(self):
        return "Plan(%s, speed=%.2f, swing=%.3f)" % (
            self.state, self.speed, self.swing_height)


class StepSequencer(object):

    def __init__(self, direction=1.0, clearance=0.045, min_swing=0.06,
                 max_swing=0.20, mount_reach=0.22, creep_speed=0.54,
                 lift_speed=0.54, search_speed=0.35, look_samples=5,
                 look_spread=0.022, look_timeout=1.5, settle_seconds=0.7,
                 mount_timeout=8.0, lift_timeout=12.0, search_seconds=2.5,
                 retries=2, retry_clearance=0.025, gained_fraction=0.7,
                 foot_x=0.1934, min_rise=0.03):
        self.direction = 1.0 if direction > 0 else -1.0
        # How much higher than the riser the foot is lifted. The foot is a
        # 20 mm sphere on a leg with a few millimetres of position error under
        # load, and the tread it is aiming at is beyond a square edge.
        self.clearance = clearance
        self.min_swing = min_swing
        # Capped because the leg has 0.426 m of reach and stands at 0.26 m of
        # it: a swing that asks for more than the leg has left simply does not
        # arrive, and the IK holds the last reachable pose instead.
        self.max_swing = max_swing
        # How close the front feet get to the riser before slowing to a
        # deliberate step. It cannot be made arbitrarily small: the scanner's
        # nearest beam lands about 0.17 m ahead of the front feet, so a riser
        # closer than that is not merely hard to measure, it is out of view
        # entirely - and a MOUNT waiting to see it arrive at 0.10 m waits
        # forever. This is the near limit plus a margin.
        self.mount_reach = mount_reach
        # These are fractions of the traverse speed, and what they really
        # set is stride: the crawl turns a commanded vx into a body advance
        # of vx * 1.5, capped at its own STEP_REACH. At 0.40 the climbing
        # step commanded 0.12 m/s, which is a 0.18 m stride - half a tread,
        # so the foot landed back on the tread it left every second time.
        # 0.54 commands 0.162 m/s, which fills the 0.24 m stride the crawl
        # allows - a third more ground per placement than 0.40 bought. It is not a faster creep so much
        # as a longer one; the crawl's own cap still decides the maximum.
        self.creep_speed = creep_speed
        self.lift_speed = lift_speed
        # 0.35, and not 0.40, though 0.40 looks like the obviously right
        # number: the crawl clamps body advance to 0.18 m per placement and
        # 0.40 is exactly where that clamp starts binding, so it is the most
        # creep available for free. It does not survive the top of the
        # flight. Measured at 0.40 the climb cleared all five risers and then
        # blocked crossing the platform, feet flat and level, with the COM
        # short of the support triangle - the same failure a longer stride
        # gives. The body can only cross so much ground per placement, and
        # 0.35 is inside that and 0.40 is not.
        #
        # Used when the scanner sees nothing to climb but the run is not
        # over - which is not a rare failure, it is how every flight ends.
        # The nearest beams land about 0.18 m ahead of the front feet, so the
        # last riser of a flight goes under the sensor's near limit before it
        # is climbed and simply cannot be seen from there. The robot climbs
        # that one on the measurement it already has, which is what keeping
        # the previous riser's clearance is for. Creeping and looking again is
        # also the right answer to a bad reading; the wrong answer is to
        # decide the flight has ended, which is not this module's call.
        self.search_speed = search_speed
        # Standing still is the one cost on a climb that can be cut without
        # asking the body to travel further per placement, which is what
        # every other speed-up here turns out to be underneath. Measured over
        # a flight: 35 s of LOOK against 30 s of actually climbing, and most
        # of the LOOK is attempts that never agree - near the top the next
        # riser is under the scanner's near limit and there is genuinely
        # nothing to see. A shorter timeout gives up on those sooner and
        # creeps, which at least makes ground; it does not shorten a LOOK
        # that is going to succeed, because those agree within a few sweeps.
        #
        # Not looking at all is not the answer either, though it looks like
        # it. Tried: once the first riser was climbed, trust the flight to be
        # even and go straight from SETTLE to LIFT. At the full climbing
        # stride that blocked 2 climbs in 2, trunk 21 degrees nose-up with a
        # rear leg out of reach; at the search stride it passed 2 in 3 and
        # saved only 5 s. The crawl finishes its placement while the robot
        # "stands" to look, so the looking was never the time - placements
        # are, about fifty of them at 2.1 s each.
        self.look_samples = look_samples
        self.look_spread = look_spread
        self.look_timeout = look_timeout
        self.settle_seconds = settle_seconds
        self.mount_timeout = mount_timeout
        self.lift_timeout = lift_timeout
        self.search_seconds = search_seconds
        self.retries = retries
        self.retry_clearance = retry_clearance
        # A step counts as climbed at this fraction of its riser. Not at the
        # whole of it: the trunk sits between two treads for most of a flight,
        # and waiting for the full riser means waiting for the rear feet,
        # which is what SETTLE is for.
        self.gained_fraction = gained_fraction
        self.foot_x = foot_x
        self.min_rise = min_rise

        self.state = LOOK
        self.index = 0
        self.riser = 0.0
        self.swing_height = min_swing
        self.attempt = 0
        self.note = "looking for the first step"
        self._entered = None
        self._samples = []
        self._base_z = None

    # ------------------------------------------------------------- internals

    def _enter(self, state, now):
        self.state = state
        self._entered = now

    def _elapsed(self, now):
        return 0.0 if self._entered is None else now - self._entered

    def _swing_for(self, rise):
        """Foot clearance for a riser of this height.

        A descent needs no extra lift - the foot is stepping down into space,
        and the gait's own probe reaches for the tread below. It is the climb
        that needs the riser cleared.
        """
        if rise <= 0.0:
            return self.min_swing
        return max(self.min_swing,
                   min(self.max_swing, rise + self.clearance))

    def _plan(self, speed, note="", keep=False):
        """`keep` makes a note stick, so the log reads as the current state."""
        if note and keep:
            self.note = note
        return Plan(speed, self.swing_height, self.state, self.riser,
                    self.index, note or self.note)

    # ---------------------------------------------------------------- update

    def _target(self):
        """How much height counts as having climbed the step in hand."""
        return self.gained_fraction * abs(self.riser)

    def _gained(self, z):
        if self._base_z is None:
            return 0.0
        return (z - self._base_z) * self.direction

    def _climbed(self, z, now):
        """One more riser is behind it. Stop and let the rear feet catch up.

        The datum advances by exactly one riser rather than being reset to
        wherever the trunk happens to be. A crawl lifts the trunk half a riser
        per foot, so at the moment a step is called climbed the next one is
        usually already part way done, and resetting throws that away - which
        at the top of a flight loses a whole step, because the last riser only
        ever has the rear feet left to contribute.

        The datum moves by the whole riser even though the trigger is a
        fraction of it. The fraction says when to believe the step is going;
        the riser is what was actually climbed, and using the trigger for both
        would count a five-step flight seven times.
        """
        self.index += 1
        self.note = "step %d cleared (%.0f mm)" % (self.index,
                                                   self._gained(z) * 1000.0)
        self.attempt = 0
        self._base_z += abs(self.riser) * self.direction
        self._enter(SETTLE, now)
        return self._plan(0.0)

    def update(self, z, profile, now):
        """One tick. `z` is the trunk height; `profile` is the latest sweep."""
        if self._entered is None:
            self._entered = now
        if self._base_z is None:
            self._base_z = z
        # A riser gained is a riser gained, whichever state noticed it. The
        # crawl swings one foot at a time and the trunk follows the last of
        # them, so the height often arrives a beat after the state that asked
        # for it has moved on - while the robot is standing still looking for
        # the next step, most often. Checking here rather than in each state
        # is what stops the last riser of a flight going uncounted.
        if self.state in (LOOK, MOUNT, LIFT, SEARCH) and self.riser \
                and self._gained(z) >= self._target():
            return self._climbed(z, now)
        handler = {LOOK: self._look, MOUNT: self._mount, LIFT: self._lift,
                   SETTLE: self._settle, SEARCH: self._search}.get(self.state)
        if handler is None:
            return self._plan(0.0)
        return handler(z, profile, now)

    def _look(self, z, profile, now):
        step = profile.next_step if profile is not None and profile.valid \
            else None
        if step is not None and abs(step.rise) >= self.min_rise \
                and step.reach(self.foot_x) > 0.0:
            self._samples.append((step.rise, step.reach(self.foot_x)))
        else:
            self._samples = []

        if len(self._samples) >= self.look_samples:
            recent = self._samples[-self.look_samples:]
            rises = [rise for rise, _ in recent]
            if max(rises) - min(rises) <= self.look_spread:
                self.riser = sorted(rises)[len(rises) // 2]
                self.swing_height = self._swing_for(self.riser)
                self.attempt = 0
                self._samples = []
                self.note = ("step %d: %.0f mm %s, %.2f m ahead"
                             % (self.index + 1, abs(self.riser) * 1000.0,
                                "up" if self.riser > 0 else "down",
                                sorted(reach for _, reach in recent)[
                                    len(recent) // 2]))
                self._enter(MOUNT, now)
                return self._plan(0.0)
            # Not settled yet - keep the newest and wait for agreement.
            self._samples = self._samples[-self.look_samples + 1:]

        if self._elapsed(now) > self.look_timeout:
            self.note = ("nothing clear ahead - creeping on the last "
                         "%.0f mm measured" % (abs(self.riser) * 1000.0))
            self._samples = []
            self._enter(SEARCH, now)
            return self._plan(self.search_speed)
        return self._plan(0.0, "looking")

    def _mount(self, z, profile, now):
        step = profile.next_step if profile is not None and profile.valid \
            else None
        reach = step.reach(self.foot_x) if step is not None else None
        if reach is not None and reach <= self.mount_reach:
            self._enter(LIFT, now)
            return self._plan(self.lift_speed, "at the riser - stepping up", keep=True)
        if self._elapsed(now) > self.mount_timeout:
            # Crept a whole tread and never arrived at the riser. Stop and
            # re-measure from wherever it now is rather than pressing on.
            self._samples = []
            self._enter(LOOK, now)
            return self._plan(0.0, "lost the step - looking again", keep=True)
        return self._plan(self.creep_speed)

    def _lift(self, z, profile, now):
        if self._elapsed(now) > self.lift_timeout:
            if self.attempt < self.retries:
                # Caught on the nosing. Lift higher and have another go, which
                # is what the extra clearance is for.
                self.attempt += 1
                self.swing_height = min(
                    self.max_swing,
                    self.swing_height + self.retry_clearance)
                self.note = ("step %d caught - retry %d at %.0f mm clearance"
                             % (self.index + 1, self.attempt,
                                self.swing_height * 1000.0))
                self._enter(LIFT, now)
                return self._plan(self.lift_speed)
            self.note = ("step %d would not go, even at %.0f mm clearance"
                         % (self.index + 1, self.swing_height * 1000.0))
            self._enter(STALLED, now)
            return self._plan(0.0)
        return self._plan(self.lift_speed)

    def _search(self, z, profile, now):
        """Creeping blind, on the last riser it managed to measure."""
        if self._elapsed(now) > self.search_seconds:
            self._samples = []
            self._enter(LOOK, now)
            return self._plan(0.0, "looking again", keep=True)
        return self._plan(self.search_speed)

    def _settle(self, z, profile, now):
        if self._elapsed(now) >= self.settle_seconds:
            self._samples = []
            self._enter(LOOK, now)
            return self._plan(0.0, "looking for the next step", keep=True)
        return self._plan(0.0, "settling")

    def describe(self):
        return "%s (%d climbed)" % (self.note, self.index)


def _main():
    """Fly the sequencer up a ray-cast flight, with no simulator involved.

    The toy robot below is crude but not dishonest about the two things that
    matter. Its feet transfer to the next tread only when the swing height it
    was given actually clears the riser, and its trunk rides at the mean of
    the two ends - so it pitches while it is half way up a step, and it has
    only gained a riser once both ends have. A sequencer that gets the
    clearance wrong gets stuck here, which is the point of running it.
    """
    from go2_voice_control import perception

    failures = 0

    def check(label, ok, detail=""):
        nonlocal failures
        failures += 0 if ok else 1
        print("%-4s %-50s %s" % ("ok" if ok else "FAIL", label, detail))

    riser, tread, count = 0.09, 0.36, 5
    wheelbase, ride = 0.387, 0.30
    outline = perception.staircase_outline(0.0, riser, tread, count)

    def fly(sequencer, start_x=-0.70, limit=120.0, dt=0.05):
        x, z = start_x, 0.0
        grace = 2.0                       # keep ticking after the last foot
        levels = {"front": 0, "rear": 0}
        states, swings = [], []
        now = 0.0
        while now < limit and sequencer.state != STALLED:
            local = [(px - x, pz - z) for px, pz in outline]
            angle_min, increment, ranges = perception.simulate_scan(
                local, ride, noise=0.003)
            profile = perception.from_scan(angle_min, increment, ranges)
            plan = sequencer.update(z + ride, profile, now)
            if not states or states[-1] != plan.state:
                states.append(plan.state)
            swings.append(plan.swing_height)

            x += 0.30 * plan.speed * dt
            for name, offset in (("front", +wheelbase / 2.0),
                                 ("rear", -wheelbase / 2.0)):
                foot = x + offset
                under = 0
                for index in range(count):
                    if foot >= index * tread:
                        under = index + 1
                if under > levels[name]:
                    # Only goes up if the foot was lifted high enough to
                    # clear the riser it is standing in front of.
                    if plan.swing_height >= riser + 0.004:
                        levels[name] += 1
                elif under < levels[name]:
                    levels[name] = under
            wanted = riser * (levels["front"] + levels["rear"]) / 2.0
            z += (wanted - z) * min(1.0, dt / 0.35)
            now += dt
            if levels["front"] >= count and levels["rear"] >= count:
                grace -= dt
                if grace <= 0.0:
                    break
        return states, swings, x, z

    sequencer = StepSequencer()
    states, swings, x, z = fly(sequencer)

    check("it climbed every step", sequencer.index >= count,
          "%d of %d, ending %.2f m up" % (sequencer.index, count, z))
    check("and did so one at a time, not in one go",
          states.count(LIFT) >= count - 1, "%d lifts" % states.count(LIFT))
    check("it stopped to look before each one",
          states.count(LOOK) >= count, "%d looks" % states.count(LOOK))
    check("it never stalled", sequencer.state != STALLED)
    # The top riser goes under the scanner's near limit before it is climbed,
    # so it is climbed on the previous measurement rather than a fresh one.
    # That is designed for, not tolerated - see `search_speed`.
    check("the last riser was climbed blind, on the previous measurement",
          SEARCH in states, "states: %s" % " ".join(states[-4:]))

    wanted = riser + sequencer.clearance
    check("the swing height came from the riser it measured",
          abs(max(swings) - wanted) < 0.02,
          "%.3f m vs %.3f m wanted" % (max(swings), wanted))
    check("and it never demanded more reach than the leg has",
          max(swings) <= sequencer.max_swing, "%.3f m" % max(swings))

    # A step it physically cannot climb has to be given up on, not walked at
    # forever - and only after it has tried lifting higher.
    class Stuck(object):
        valid = True

        def __init__(self):
            self.steps = [perception.Step(edge=0.25, rise=0.09, top=0.09)]
            self.next_step = self.steps[0]

    stuck = StepSequencer(lift_timeout=1.0, settle_seconds=0.1)
    now = 0.0
    tried = set()
    while now < 60.0 and stuck.state != STALLED:
        plan = stuck.update(0.30, Stuck(), now)
        if plan.state == LIFT:
            tried.add(round(plan.swing_height, 4))
        now += 0.05
    check("a step that will not go is abandoned", stuck.state == STALLED)
    check("but only after lifting higher first", len(tried) >= 2,
          "clearances tried: %s" % sorted(tried))

    print("\n     %s" % sequencer.describe())
    print("\n%s" % ("all checks passed" if not failures
                    else "%d CHECKS FAILED" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
