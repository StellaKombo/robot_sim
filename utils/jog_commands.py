"""Reachable jogging command programs with explicit holds and transitions."""

import numpy as np


PROFILE_NAMES = ("standing", "steady", "speed_changes", "curves", "pivots",
                 "start_stop", "rapid_restart", "mode_transitions")
PROFILE_COUNTS = (10, 15, 15, 15, 10, 15, 5, 15)  # per 100 episodes
BEHAVIOR_NAMES = ("stand", "forward", "backward", "curve_left", "curve_right",
                  "pivot_left", "pivot_right")
DIRECTED_PAIRS = tuple((a, b) for a in range(7) for b in range(7) if a != b)
PROFILE_SLOTS = np.repeat(PROFILE_NAMES, PROFILE_COUNTS)


def command_behavior(command):
    command = np.asarray(command)
    vx, wz = command[..., 0], command[..., 2]
    result = np.zeros(command.shape[:-1], dtype=np.int8)
    result[(vx > 0) & (wz == 0)] = 1
    result[(vx < 0) & (wz == 0)] = 2
    result[(vx != 0) & (wz > 0)] = 3
    result[(vx != 0) & (wz < 0)] = 4
    result[(vx == 0) & (wz > 0)] = 5
    result[(vx == 0) & (wz < 0)] = 6
    return result


def validate_commands(command):
    c = np.asarray(command)
    if c.ndim != 3 or c.shape[-1] != 3 or not np.isfinite(c).all():
        raise ValueError("command schedule must be finite (batch, queries, 3)")
    vx, vy, wz = c[..., 0], c[..., 1], c[..., 2]
    stand = (vx == 0) & (wz == 0)
    drive = (((vx >= .5) & (vx <= 1.5)) | ((vx >= -1.) & (vx <= -.5))) & (np.abs(wz) <= .5)
    pivot = (vx == 0) & (np.abs(wz) >= 1.) & (np.abs(wz) <= 2.)
    if not ((vy == 0) & (stand | drive | pivot)).all():
        raise ValueError("command schedule leaves the policy's trained twist bands")
    return c


def make_jog_program(batch_size, queries, *, seed, episode_offset=0, control_dt=.02, period=43):
    """Return deterministic per-episode schedules; no robot or actor resets inside.

    Every moving episode reserves at least three seconds for the final stop.
    Smooth ramps stay within one trained band; mode changes remain explicit.
    Global episode IDs determine profiles, independently of collection batches.
    """
    if batch_size < 1 or queries*control_dt < 12. or seed < 0 or episode_offset < 0:
        raise ValueError("positive batches, nonnegative seeds/offsets, and >=12 s are required")
    commands = np.zeros((batch_size, queries, 3), dtype=np.float32)
    labels, variants = [], []
    phases = np.empty(batch_size, dtype=np.int64)

    for row in range(batch_size):
        identity = episode_offset + row
        rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
        profile = str(PROFILE_SLOTS[(37*identity + seed) % 100])
        labels.append(profile)
        phases[row] = 0 if identity % 5 == 0 else rng.integers(period)
        limit = queries - int(round(3./control_dt))
        cursor = 0

        def speed(lo, hi):
            return float(rng.choice([lo, hi])) if rng.random() < .25 else float(rng.uniform(lo, hi))

        def twist(behavior):
            if behavior == 0:
                return np.zeros(3, dtype=np.float32)
            if behavior == 1:
                return np.array([speed(.5, 1.5), 0., 0.], dtype=np.float32)
            if behavior == 2:
                return np.array([-speed(.5, 1.), 0., 0.], dtype=np.float32)
            if behavior in (3, 4):
                vx = -speed(.5, 1.) if rng.random() < .25 else speed(.5, 1.5)
                return np.array([vx, 0., (1 if behavior == 3 else -1)*speed(.08, .5)], dtype=np.float32)
            return np.array([0., 0., (1 if behavior == 5 else -1)*speed(1., 2.)], dtype=np.float32)

        def hold(value, seconds):
            nonlocal cursor
            end = min(limit, cursor + max(1, int(round(seconds/control_dt))))
            commands[row, cursor:end] = value
            cursor = end

        def ramp(before, after, seconds):
            nonlocal cursor
            end = min(limit, cursor + max(1, int(round(seconds/control_dt))))
            if end > cursor:
                alpha = np.linspace(0., 1., end-cursor, dtype=np.float32)[:, None]
                commands[row, cursor:end] = before + alpha*(after-before)
                cursor = end

        hold(twist(0), rng.uniform(.8, 1.6))
        if profile == "standing":
            variants.append("cold_and_settled_standing")
        elif profile == "steady":
            behavior = 1 + (identity//100 + identity) % 6
            hold(twist(behavior), queries*control_dt)
            variants.append(BEHAVIOR_NAMES[behavior])
        elif profile == "speed_changes":
            behavior = int(rng.choice([1, 2]))
            before = twist(behavior)
            while cursor < limit:
                hold(before, rng.uniform(1., 3.))
                after = twist(behavior)
                ramp(before, after, rng.uniform(.3, 1.2))
                before = after
            variants.append(BEHAVIOR_NAMES[behavior])
        elif profile == "curves":
            vx = speed(.5, 1.5) * (1 if identity % 4 else -1)
            if vx < -.5:
                vx = max(-1., vx)
            turn = speed(.08, .5)
            while cursor < limit:
                for wz in (0., turn, 0., -turn):
                    hold([vx, 0., wz], rng.uniform(1.2, 3.))
            variants.append("backward_curves" if vx < 0 else "forward_curves")
        elif profile == "pivots":
            while cursor < limit:
                hold(twist(5), rng.uniform(1.5, 3.))
                hold(twist(0), rng.uniform(.8, 2.))
                hold(twist(6), rng.uniform(1.5, 3.))
                hold(twist(0), rng.uniform(.8, 2.))
            variants.append("both_pivot_directions")
        elif profile in ("start_stop", "rapid_restart"):
            while cursor < limit:
                hold(twist(int(rng.integers(1, 7))), rng.uniform(1.5, 3.5))
                hold(twist(0), rng.uniform(.08, .5) if profile == "rapid_restart" else rng.uniform(1.5, 3.))
            variants.append("short_stop" if profile == "rapid_restart" else "settled_stop")
        else:
            first, second = DIRECTED_PAIRS[identity % len(DIRECTED_PAIRS)]
            a, b = twist(first), twist(second)
            while cursor < limit:
                hold(a, rng.uniform(2., 4.))
                hold(b, rng.uniform(2., 4.))
            variants.append(f"{BEHAVIOR_NAMES[first]}_to_{BEHAVIOR_NAMES[second]}")

    validate_commands(commands)
    return dict(commands=commands, behavior=command_behavior(commands),
                profile_labels=np.asarray(labels), profile_variants=np.asarray(variants),
                initial_phase=phases,
                episode_ids=np.arange(episode_offset, episode_offset+batch_size, dtype=np.int64))
