"""Smooth, feasible unicycle excitation supplement for the jogging KCF corpus."""

import json
import numpy as np

from utils.jog_commands import command_behavior, validate_commands


PROFILE_NAMES = ("figure_eight", "ovals", "slalom", "multisine", "smooth_waypoints",
                 "smooth_pivots", "smooth_restart")
PROFILE_COUNTS = (25, 15, 15, 15, 15, 5, 10)
PROFILE_SLOTS = np.repeat(PROFILE_NAMES, PROFILE_COUNTS)


def quintic(alpha):
    alpha = np.clip(alpha, 0., 1.)
    return alpha**3 * (10. + alpha*(-15. + 6.*alpha))


def integrate_unicycle(commands, dt=.02):
    """Exact planar integration of each 20 ms held body twist, initial pose zero."""
    commands = np.asarray(commands, dtype=np.float64)
    delta = commands[..., 2]*dt
    heading = np.concatenate((np.zeros(commands.shape[:-2]+(1,)), np.cumsum(delta, axis=-1)), axis=-1)
    distance = commands[..., 0]*dt*np.sinc(delta/(2.*np.pi))
    midpoint = heading[..., :-1] + .5*delta
    displacement = distance[..., None]*np.stack((np.cos(midpoint), np.sin(midpoint)), axis=-1)
    position = np.concatenate((np.zeros(commands.shape[:-2]+(1, 2)), np.cumsum(displacement, axis=-2)), axis=-2)
    return np.concatenate((position, heading[..., None]), axis=-1)


def _speed(rng, backward):
    mean = rng.uniform(.65, .84) if backward else rng.uniform(.7, 1.22)
    low, high = (.5, 1.) if backward else (.5, 1.5)
    amplitude = rng.uniform(.25, .9)*min(mean-low, high-mean)
    return (-1. if backward else 1.), mean, amplitude


def _periodic_path(rng, kind, available_s, backward):
    """Arc-length parameterization preserves forward/backward speed bands.

    For a Gerono figure-eight x=sin(theta), y=r*sin(2*theta); ellipse
    x=cos(theta), y=r*sin(theta). Scale and traversal time bound v and yaw rate
    analytically over a dense full-cycle grid, without clipping the path.
    """
    ratio = rng.uniform(.26, .33) if kind == "figure_eight" else rng.choice([1., rng.uniform(.66, .95)])
    grid = np.linspace(0., 2*np.pi, 32769)
    if kind == "figure_eight":
        dx, dy = np.cos(grid), 2*ratio*np.cos(2*grid)
        ddx, ddy = -np.sin(grid), -4*ratio*np.sin(2*grid)
    else:
        dx, dy = -np.sin(grid), ratio*np.cos(grid)
        ddx, ddy = -np.cos(grid), -ratio*np.sin(grid)
    metric = np.hypot(dx, dy)
    arc = np.r_[0., np.cumsum(.5*(metric[1:]+metric[:-1])*np.diff(grid))]
    length = arc[-1]
    curvature_max = np.max(np.abs((dx*ddy-dy*ddx)/metric**3))
    sign, mean, _ = _speed(rng, backward)
    modulation = rng.uniform(.04, .12)
    minimum_period = length*curvature_max*(1.+modulation)/.47
    low = max(48. if kind == "figure_eight" else 30., minimum_period)
    high = available_s-.12
    if low >= high:
        raise ValueError("Episode is too short for a complete feasible path")
    period = rng.uniform(low, high)
    scale = mean*period/length
    phase = rng.uniform(-np.pi, np.pi)
    offset = rng.uniform(0., length)
    turn_sign = float(rng.choice([-1., 1.]))

    def sample(time):
        time = np.asarray(time)
        a = 2*np.pi*time/period + phase
        traveled = length*(time/period + modulation/(2*np.pi)*(np.cos(phase)-np.cos(a)))
        theta = np.interp(np.remainder(offset+turn_sign*traveled, length), arc, grid)
        if kind == "figure_eight":
            tx, ty = np.cos(theta), 2*ratio*np.cos(2*theta)
            ax, ay = -np.sin(theta), -4*ratio*np.sin(2*theta)
        else:
            tx, ty = -np.sin(theta), ratio*np.cos(theta)
            ax, ay = -np.cos(theta), -ratio*np.sin(theta)
        curvature = (tx*ay-ty*ax)/np.hypot(tx, ty)**3
        speed = mean*(1.+modulation*np.sin(a))
        return np.stack((sign*speed, np.zeros_like(speed), turn_sign*speed*curvature/scale), axis=-1)

    metadata = dict(path=kind, aspect_ratio=float(ratio), spatial_scale_m=float(scale),
                    period_s=float(period), mean_speed_m_s=float(mean), speed_modulation=float(modulation),
                    arc_offset=float(offset), speed_phase=float(phase), traversal_sign=turn_sign,
                    backward=bool(backward))
    return sample, metadata


def _driving_signal(rng, kind, backward, duration_s):
    sign, mean, amplitude = _speed(rng, backward)
    if kind == "slalom":
        frequency = rng.uniform(1./24., 1./10.)
        phase = rng.uniform(-np.pi, np.pi, 2)
        turn = rng.uniform(.15, .47)

        def sample(time):
            time = np.asarray(time)
            vx = sign*(mean+amplitude*np.sin(2*np.pi*frequency*time+phase[0]))
            wz = turn*np.sin(2*np.pi*frequency*time+phase[1])
            return np.stack((vx, np.zeros_like(vx), wz), axis=-1)

        metadata = dict(path=kind, frequency_hz=float(frequency), yaw_amplitude=float(turn),
                        phase=phase.tolist())
    elif kind == "multisine":
        frequency = np.stack([rng.uniform(a, b, 2) for a, b in ((.02, .07), (.07, .16), (.16, .30))])
        phase = rng.uniform(-np.pi, np.pi, (3, 2))
        weights = np.array([.55, .3, .15])
        turn = rng.uniform(.2, .47)

        def sample(time):
            time = np.asarray(time)
            wave = np.sum(weights[:, None]*np.sin(2*np.pi*time[..., None, None]*frequency+phase), axis=-2)
            vx = sign*(mean+amplitude*wave[..., 0])
            return np.stack((vx, np.zeros_like(vx), turn*wave[..., 1]), axis=-1)

        metadata = dict(path=kind, frequency_hz=frequency.tolist(), phase=phase.tolist(),
                        yaw_amplitude=float(turn), weights=weights.tolist())
    else:
        knots = [0.]
        while knots[-1] < duration_s+3.:
            knots.append(knots[-1]+rng.uniform(1.2, 4.))
        knots = np.asarray(knots)
        low, high = (.51, .99) if backward else (.51, 1.49)
        values = np.stack((sign*rng.uniform(low, high, len(knots)), np.zeros(len(knots)),
                           rng.uniform(-.47, .47, len(knots))), axis=-1)

        def sample(time):
            time = np.asarray(time)
            index = np.clip(np.searchsorted(knots, time, side="right")-1, 0, len(knots)-2)
            alpha = quintic((time-knots[index])/(knots[index+1]-knots[index]))
            return values[index] + alpha[..., None]*(values[index+1]-values[index])

        metadata = dict(path=kind, knot_time_s=knots.tolist(), knot_commands=values.tolist())
    metadata.update(backward=bool(backward), mean_speed_m_s=float(mean), speed_amplitude=float(amplitude))
    return sample, metadata


def _drive_segment(rng, kind, count, dt, backward):
    """Straight lead/tail and C2 blends into a smooth driving signal."""
    time = np.arange(count)*dt
    lead, blend = .30, .70
    core_start = lead+blend
    core_end = count*dt-core_start
    if kind in ("figure_eight", "ovals"):
        sample, metadata = _periodic_path(rng, kind, core_end-core_start, backward)
    else:
        sample, metadata = _driving_signal(rng, kind, backward, count*dt)
    centered = time-core_start
    commands = sample(centered)
    before = np.array([sample(np.array(-blend))[0], 0., 0.])
    after = np.array([sample(np.array(core_end-core_start+blend))[0], 0., 0.])
    ingress = time < core_start
    weight = quintic((time[ingress]-lead)/blend)
    commands[ingress] = before + weight[:, None]*(commands[ingress]-before)
    egress = time >= core_end
    weight = quintic((time[egress]-core_end)/blend)
    commands[egress] += weight[:, None]*(after-commands[egress])
    metadata.update(core_start_s=core_start, core_end_s=core_end, lead_s=lead, blend_s=blend)
    return commands.astype(np.float32), metadata


def make_smooth_jog_program(batch_size, queries, *, seed, episode_offset=2_000_000,
                            control_dt=.02, period=43):
    """Return reproducible smooth profiles; mode switches retain deployed gaps."""
    if batch_size < 1 or queries*control_dt < 60. or seed < 0 or episode_offset < 0:
        raise ValueError("Smooth collections require positive batches and at least 60 seconds")
    commands = np.zeros((batch_size, queries, 3), dtype=np.float32)
    phases = np.empty(batch_size, dtype=np.int64)
    labels, variants, parameters = [], [], []
    core_start = np.full(batch_size, -1, dtype=np.int64)
    core_end = np.full(batch_size, -1, dtype=np.int64)
    periods = np.zeros(batch_size, dtype=np.float64)
    for row in range(batch_size):
        identity = episode_offset+row
        rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
        profile = str(PROFILE_SLOTS[(37*identity+seed) % 100])
        phases[row] = 0 if identity % 5 == 0 else rng.integers(period)
        labels.append(profile)
        backward = bool(rng.random() < .3)
        start = int(round(rng.uniform(.8, 1.5)/control_dt))
        end = queries-int(round(3./control_dt))
        if profile == "smooth_pivots":
            cursor = start
            segments = []
            direction = int(rng.choice([-1, 1]))
            while cursor < end:
                count = min(end-cursor, int(round(rng.uniform(5., 9.)/control_dt)))
                time = np.arange(count)*control_dt
                mean, amplitude, frequency, phase = rng.uniform(1.35, 1.65), rng.uniform(.15, .3), rng.uniform(.05, .2), rng.uniform(-np.pi, np.pi)
                commands[row, cursor:cursor+count, 2] = direction*(mean+amplitude*np.sin(2*np.pi*frequency*time+phase))
                segments.append(dict(start_query=cursor, end_query=cursor+count, direction=direction,
                                     mean=float(mean), amplitude=float(amplitude), frequency_hz=float(frequency), phase=float(phase)))
                cursor += count+int(round(rng.uniform(1., 2.)/control_dt))
                direction *= -1
            metadata = dict(path=profile, segments=segments)
            variant = "both_directions_with_stops"
        elif profile == "smooth_restart":
            cursor, segments = start, []
            while cursor < end:
                count = min(end-cursor, int(round(rng.uniform(5., 10.)/control_dt)))
                kind = str(rng.choice(["slalom", "multisine", "smooth_waypoints"]))
                backward = bool(rng.random() < .35)
                values, metadata = _drive_segment(rng, kind, count, control_dt, backward)
                commands[row, cursor:cursor+count] = values
                segments.append(dict(start_query=cursor, end_query=cursor+count, **metadata))
                stop_seconds = rng.uniform(.08, .5) if rng.random() < .5 else rng.uniform(1.5, 3.)
                cursor += count+max(1, int(round(stop_seconds/control_dt)))
            metadata = dict(path=profile, segments=segments)
            variant = "smooth_segments_with_settled_and_early_restarts"
        else:
            values, metadata = _drive_segment(rng, profile, end-start, control_dt, backward)
            commands[row, start:end] = values
            core_start[row] = start+int(round(metadata["core_start_s"]/control_dt))
            core_end[row] = start+int(round(metadata["core_end_s"]/control_dt))
            periods[row] = metadata.get("period_s", 0.)
            variant = "backward" if backward else "forward"
            if profile in ("figure_eight", "ovals"):
                variant += "_left_first" if metadata["traversal_sign"] > 0 else "_right_first"
        variants.append(variant)
        parameters.append(json.dumps(metadata, sort_keys=True, allow_nan=False))
    validate_commands(commands)
    return dict(commands=commands, behavior=command_behavior(commands), profile_labels=np.asarray(labels),
                profile_variants=np.asarray(variants), initial_phase=phases,
                episode_ids=np.arange(episode_offset, episode_offset+batch_size, dtype=np.int64),
                profile_parameters=np.asarray(parameters), path_core_start_query=core_start,
                path_core_end_query=core_end, path_period_s=periods,
                command_integrated_reference=integrate_unicycle(commands, control_dt))
