"""Encode live MuJoCo state using the existing 84-D G1 KCF schema.
"""

from typing import Protocol

import numpy as np

from utils.math_utils import gravity_from_quat, q_to_rom, v_to_rom, world_to_body


class G1State(Protocol):
    """State fields needed by the encoder (MuJoCo qpos/qvel conventions)."""

    t: float
    q: np.ndarray
    v: np.ndarray
    prev_action: np.ndarray


def g1_closed_loop_state(state: G1State, gait_period: float,
                         theta_prev: float | None = None) -> np.ndarray:
    """84-D Markov state of the full-order KCF model, as in the training data.

    [px, py, theta] (3), [vx, vy, wz] (3), pz (1), gravity (3), joint q (23),
    [vz, wx, wy] (3), joint v (23), gait [sin, cos] (2), previous action (23).
    Pass the previous tick's theta to unwrap yaw across consecutive states.
    """
    q = state.q.astype(np.float32)
    v = state.v.astype(np.float32)
    quat = q[3:7]
    planar_pose = q_to_rom(q).astype(np.float32, copy=False)
    if theta_prev is not None:
        planar_pose[2] = np.unwrap(np.array([theta_prev, planar_pose[2]], dtype=np.float32))[1]
    planar_velocity = v_to_rom(q, v).astype(np.float32, copy=False)
    planar_velocity[2] = v[5]
    gravity = gravity_from_quat(quat)
    linear_body = world_to_body(quat, v[:3])
    angular_body = v[3:6]
    remaining_base_velocity = np.array(
        [linear_body[2], angular_body[0], angular_body[1]], dtype=np.float32
    )
    phase = np.remainder(state.t, gait_period) / gait_period
    gait = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)
    return np.concatenate([planar_pose, planar_velocity, q[2:3], gravity, q[7:],
                           remaining_base_velocity, v[6:], gait, state.prev_action.astype(np.float32)])
