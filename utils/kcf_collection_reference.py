"""Bounded, deterministic reference families for offline MPC data collection.

This module does not change the production fixed/Lissajous reference defaults.
Pose schedules use quintic interpolation: velocity and acceleration are zero
at every waypoint, providing explicit stops rather than discontinuous jumps.
"""

from copy import deepcopy
import numpy as np
from utils.kcf_reference import PlanarReference

MODES = ("figure8", "standing", "stop_start", "waypoints")


class CollectionReference(PlanarReference):
    def __init__(self, settings, specification):
        super().__init__(settings)
        self.specification = deepcopy(specification)
        self.mode = specification["mode"]
        if self.mode not in MODES:
            raise ValueError("Unknown collection reference mode")
        if self.mode == "figure8":
            values = [specification[key] for key in ("omega", "half_width", "half_height", "rotation", "phase")]
            if (not np.isfinite(values).all() or min(values[:3]) <= 0
                    or specification["direction"] not in (-1, 1)
                    or np.shape(specification["origin"]) != (2,)
                    or not np.isfinite(specification["origin"]).all()):
                raise ValueError("Invalid figure-eight dimensions, frequency, direction or origin")
        if self.mode != "figure8":
            self.knots = np.asarray(specification["times"], dtype=float)
            self.poses = np.asarray(specification["poses"], dtype=float)
            if (self.knots.ndim != 1 or len(self.knots) < 2 or self.knots[0] != 0
                    or self.poses.shape != (len(self.knots), 3)
                    or not np.isfinite(self.knots).all() or not np.isfinite(self.poses).all()
                    or np.any(np.diff(self.knots) <= 0)):
                raise ValueError("Reference requires finite, increasing times and matching poses")
            # Unwrap only the public diagnostic yaw schedule; cost still uses sin/cos.
            self.poses[:, 2] = np.unwrap(self.poses[:, 2])

    def reference(self, t):
        t = np.asarray(t, dtype=float)
        if self.mode == "figure8":
            s = self.specification
            w, direction = float(s["omega"]), int(s["direction"])
            phase = direction * w * t + float(s["phase"])
            a, b = float(s["half_width"]), float(s["half_height"])
            px, py = a * np.sin(phase), b * np.sin(2 * phase)
            vx, vy = direction * a*w*np.cos(phase), direction * 2*b*w*np.cos(2*phase)
            ax, ay = -a*w*w*np.sin(phase), -4*b*w*w*np.sin(2*phase)
            angle = float(s["rotation"])
            c, sn = np.cos(angle), np.sin(angle)
            x, y = c*px-sn*py, sn*px+c*py
            dx, dy = c*vx-sn*vy, sn*vx+c*vy
            theta = np.arctan2(dy, dx)
            wz = (vx*ay-vy*ax)/(vx*vx+vy*vy)
            origin = np.asarray(s["origin"])
            return np.stack((x+origin[0], y+origin[1], theta), -1), np.stack((dx, dy, wz), -1)
        index = np.clip(np.searchsorted(self.knots, t, side="right")-1, 0, len(self.knots)-2)
        length = self.knots[index+1] - self.knots[index]
        tau = np.clip((t-self.knots[index])/length, 0., 1.)
        blend = 10*tau**3 - 15*tau**4 + 6*tau**5
        rate = 30*tau**2*(1-tau)**2/length
        change = self.poses[index+1] - self.poses[index]
        return self.poses[index] + blend[..., None]*change, rate[..., None]*change


def make_specification(mode, seed, duration):
    if mode not in MODES or not np.isfinite(duration) or duration < .1:
        raise ValueError("Expected a known mode and duration >= 0.1 s")
    rng = np.random.default_rng(seed)
    origin, yaw = rng.uniform(-1., 1., 2), float(rng.uniform(-np.pi, np.pi))
    if mode == "figure8":
        return dict(mode=mode, origin=origin.tolist(), rotation=yaw,
                    half_width=float(rng.uniform(.9, 1.5)), half_height=float(rng.uniform(.45, .75)),
                    omega=float(rng.uniform(.12, .25)), direction=int(rng.choice([-1, 1])),
                    phase=float(rng.uniform(-np.pi, np.pi)))
    pose = np.r_[origin, yaw]
    if mode == "standing":
        return dict(mode=mode, times=[0., duration], poses=[pose.tolist(), pose.tolist()])
    # Durations below 12 s are intended for smoke tests only; scale displacement
    # with time so a short test does not ask for impossible reference speeds.
    scale = min(1., duration/20.)
    times, poses = [0.], [pose.copy()]
    if mode == "stop_start":
        # Translate with independent heading to cover forward, lateral and
        # backward body commands across seeds. Every middle section is a dwell.
        displacement = rng.uniform(.35, .7)*scale * np.array([np.cos(yaw), np.sin(yaw)])
        displacement *= rng.choice([-1., 1.])
        other = pose.copy()
        other[:2] += displacement
        final = other.copy()
        final[:2] += np.array([-displacement[1], displacement[0]])
        times = (duration*np.array([0., .1, .35, .60, .85, 1.])).tolist()
        poses = [pose, pose, other, other, final, final]
    else:
        for fraction in (.25, .50, .75, 1.):
            change = rng.uniform(-.45, .45, 2)*scale
            target = poses[-1].copy()
            target[:2] += change
            target[2] += float(rng.uniform(-.45, .45))*scale
            # Three seconds moving + two seconds holding in a 20 s episode.
            times.extend([duration*(fraction-.10), duration*fraction])
            poses.extend([target, target.copy()])
    return dict(mode=mode, times=times, poses=np.asarray(poses).tolist())
