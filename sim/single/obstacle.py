"""MuJoCo mocap sphere and its configuration."""

from dataclasses import dataclass
import math
from numbers import Real
import mujoco
import numpy as np


OBSTACLE_BODY = 'mpc_obstacle'
OBSTACLE_GEOM = 'mpc_obstacle_sphere'


@dataclass(frozen=True)
class MocapSphere:
    enabled: bool = True
    radius: float = 0.07
    position: tuple[float, float, float] = (0.7, -0.4, 0.15)
    rgba: tuple[float, float, float, float] = (0.9, 0.08, 0.08, 1.0)
    collisions: bool = True
    velocity: tuple[float, float, float] = (0., 0., 0.)
    velocity_changes: tuple = ()  # (simulation time, world XYZ velocity), piecewise constant
    repeat_period: float | None = None  # Repeat a closed schedule indefinitely.

    # Define the color and position of the obstacle
    def __post_init__(self):
        if not isinstance(self.enabled, bool) or not isinstance(self.collisions, bool):
            raise ValueError('Sphere enabled/collisions must be boolean')
        if (isinstance(self.radius, bool) or not isinstance(self.radius, Real)
                or not math.isfinite(self.radius) or self.radius <= 0):
            raise ValueError('Sphere radius must be finite and positive')

        position = np.asarray(self.position, dtype=float)
        color = np.asarray(self.rgba, dtype=float)

        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Sphere position must contain three finite world coordinates')
        if (color.shape != (4,) or not np.isfinite(color).all()
                or np.any(color < 0) or np.any(color > 1)):
            raise ValueError('Sphere RGBA must contain four values between zero and one')

        object.__setattr__(self, 'position', tuple(position))
        object.__setattr__(self, 'rgba', tuple(color))
        velocity = np.asarray(self.velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            raise ValueError('Obstacle velocity must contain three finite world coordinates')
        changes, previous_time = [], 0.
        for stamp, value in self.velocity_changes:
            value = np.asarray(value, dtype=float)
            if (not math.isfinite(stamp) or stamp <= previous_time
                    or value.shape != (3,) or not np.isfinite(value).all()):
                raise ValueError('Velocity changes need increasing positive times and finite XYZ velocities')
            changes.append((float(stamp), tuple(value)))
            previous_time = stamp
        object.__setattr__(self, 'velocity', tuple(velocity))
        object.__setattr__(self, 'velocity_changes', tuple(changes))
        if self.repeat_period is not None:
            period = self.repeat_period
            if (isinstance(period, bool) or not isinstance(period, Real)
                    or not math.isfinite(period) or period <= previous_time):
                raise ValueError('Repeat period must be finite, positive and after the last velocity change')
            displacement, previous_time, current = np.zeros(3), 0., velocity
            for stamp, next_velocity in changes:
                displacement += (stamp - previous_time) * current
                previous_time, current = stamp, np.asarray(next_velocity)
            displacement += (period - previous_time) * current
            if not np.allclose(displacement, 0., rtol=0., atol=1e-9):
                raise ValueError('Repeating motion must return to its initial position without teleporting')
            object.__setattr__(self, 'repeat_period', float(period))

    @classmethod
    def crossing(cls, *, speed, center, half_extent, **sphere_settings):
        """Continuous left/right, then top/bottom passes; speed is in m/s.

        Start at the left endpoint. Return through the crossing centre before
        switching axes, then repeat for any reference duration.
        """
        center, extent = np.asarray(center, dtype=float), np.asarray(half_extent, dtype=float)
        if (isinstance(speed, bool) or not isinstance(speed, Real)
                or not math.isfinite(speed) or speed <= 0.):
            raise ValueError('Crossing speed must be finite and positive')
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Crossing center must contain three finite world coordinates')
        if extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0.):
            raise ValueError('Crossing half_extent must contain two finite positive distances')
        x, y = extent
        offsets = np.array([[-x, 0., 0.], [x, 0., 0.], [0., 0., 0.],
                            [0., y, 0.], [0., -y, 0.], [0., 0., 0.], [-x, 0., 0.]])
        legs = np.diff(offsets, axis=0)
        lengths = np.linalg.norm(legs, axis=1)
        times = np.cumsum(lengths / speed)
        velocities = legs * (speed / lengths[:, None])
        return cls(position=tuple(center + offsets[0]), velocity=tuple(velocities[0]),
                   velocity_changes=tuple(zip(times[:-1], velocities[1:])),
                   repeat_period=float(times[-1]), **sphere_settings)

    @property
    def moving(self):
        return any(self.velocity) or bool(self.velocity_changes)

    def sample(self, stamp):
        """Continuous scripted position; the controller never sees this schedule."""
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError('Obstacle sample time must be finite and nonnegative')
        if self.repeat_period is not None:
            stamp %= self.repeat_period
        position, velocity, previous_time = np.array(self.position), np.array(self.velocity), 0.
        for change_time, next_velocity in self.velocity_changes:
            if stamp < change_time:
                break
            position += (change_time - previous_time) * velocity
            velocity, previous_time = np.asarray(next_velocity), change_time
        return position + (stamp - previous_time) * velocity, velocity.copy()

    def add_to(self, spec):
        if not self.enabled:
            return
        body = spec.worldbody.add_body(name=OBSTACLE_BODY, mocap=True, pos=self.position)
        body.add_geom(name=OBSTACLE_GEOM, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[self.radius, 0, 0], rgba=self.rgba, group=0, density=0,
                      contype=int(self.collisions), conaffinity=int(self.collisions))


DEFAULT_OBSTACLE = MocapSphere()

# Shared moving environment for every reference and both predictors. These are
# sphere-center travel limits; the default sphere stays inside the 5 x 5 m box.
CROSSING_MOTION = dict(speed=1.0, center=(0., 0., .15), half_extent=(2.2, 2.2))
