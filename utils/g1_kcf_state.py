"""Encode live MuJoCo feedback in the trained walking/jogging KCF schemas."""

from typing import Protocol
import numpy as np
from utils.math_utils import gravity_from_quat, q_to_rom, v_to_rom, world_to_body


class G1State(Protocol):
    """State fields needed by the encoder (MuJoCo qpos/qvel conventions)."""

    t: float
    q: np.ndarray
    v: np.ndarray
    prev_action: np.ndarray


def g1_closed_loop_state(state: G1State, gait_period: float) -> np.ndarray:
    """Build the native state, with position/heading in world coordinates.

    [px, py, sin(theta), cos(theta)] (4), [vx, vy, wz] (3), pz (1),
    gravity (3), joint q (23),
    [vz, wx, wy] (3), joint v (23), gait [sin, cos] (2), previous action (23).
    Jogging uses 29 joints, its actual ungated clock, and 128+128 LSTM inputs,
    giving 359 channels. MPC anchors either complete vector to one rollout frame.
    """
    q = state.q.astype(np.float32)
    v = state.v.astype(np.float32)
    joints = len(state.prev_action)
    if joints not in (23, 29) or q.shape != (7+joints,) or v.shape != (6+joints,):
        raise ValueError('Robot q/v/action sizes must match the 23- or 29-DoF schema')
    quat = q[3:7]
    planar_pose = q_to_rom(q).astype(np.float32, copy=False)
    pose = np.array(
        [
            planar_pose[0],
            planar_pose[1],
            np.sin(planar_pose[2]),
            np.cos(planar_pose[2]),
        ],
        dtype=np.float32,
    )
    planar_velocity = v_to_rom(q, v).astype(np.float32, copy=False)
    planar_velocity[2] = v[5]
    gravity = gravity_from_quat(quat)
    linear_body = world_to_body(quat, v[:3])
    angular_body = v[3:6]
    remaining_base_velocity = np.array(
        [linear_body[2], angular_body[0], angular_body[1]], dtype=np.float32
    )
    phase = np.remainder(state.t, gait_period) / gait_period
    gait = np.array(
        [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32
    )
    memory = []
    if joints == 29:
        for name, width in (('policy_clock', 2), ('policy_h', 128), ('policy_c', 128)):
            value = np.asarray(getattr(state, name, None), dtype=np.float32)
            if value.shape != (width,) or not np.isfinite(value).all():
                raise ValueError(f'Jogging feedback requires actual finite {name} with {width} values')
            if name == 'policy_clock':
                if not np.isclose(np.linalg.norm(value), 1., atol=1e-5):
                    raise ValueError('Jogging clock must be the ungated unit sine/cosine pair')
                gait = value
            else:
                memory.append(value)
    return np.concatenate(
        [
            pose,
            planar_velocity,
            q[2:3],
            gravity,
            q[7:],
            remaining_base_velocity,
            v[6:],
            gait,
            state.prev_action.astype(np.float32),
            *memory,
        ]
    )


def g1_world_heading(state: G1State, previous: float | None = None) -> float:
    """Continuous measured heading for diagnostic metadata and frame anchoring."""
    heading = float(q_to_rom(state.q.astype(np.float32))[2])
    if previous is not None:
        heading = previous + np.arctan2(
            np.sin(heading - previous), np.cos(heading - previous)
        )
    return float(heading)
