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

    @property
    def moving(self):
        return any(self.velocity) or bool(self.velocity_changes)

    def sample(self, stamp):
        """Continuous scripted position; the controller never sees this schedule."""
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError('Obstacle sample time must be finite and nonnegative')
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
