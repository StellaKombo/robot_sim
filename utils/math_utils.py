##
#
# Useful math utilities for the G1 humanoid simulation and policy.
#
##

import numpy as np
import torch


def gravity_from_quat(quat: np.ndarray) -> np.ndarray:
    """Projected gravity in body frame from base quaternion (w,x,y,z)."""
    qw, qx, qy, qz = quat
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return np.array([gx, gy, gz], dtype=np.float32)


def get_gravity_orientation_batch(quat: torch.Tensor) -> torch.Tensor:
    """Projected gravity in body frame from base quaternion (w,x,y,z).
    Batched torch version of `gravity_from_quat`. Input (B,4), output (B,3).
    """
    qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    gx = 2.0 * (-qz * qx + qw * qy)
    gy = -2.0 * (qz * qy + qw * qx)
    gz = 1.0 - 2.0 * (qw * qw + qz * qz)
    return torch.stack([gx, gy, gz], dim=-1)


def body_to_world(quat: np.ndarray, v_body: np.ndarray) -> np.ndarray:
    """Rotate a vector from body to world frame: v_world = R(q) @ v_body.

    Uses the closed-form  R(q) v = v + 2w (q_vec x v) + 2 q_vec x (q_vec x v).
    Broadcasts over arbitrary leading dims.

    Args:
        quat:   (..., 4) unit quaternion in (w, x, y, z) order.
        v_body: (..., 3) vector(s) expressed in the body frame.
    Returns:
        v_world: (..., 3) same vector(s) expressed in the world frame.
    """
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    c = np.cross(q_vec, v_body, axis=-1)
    return v_body + 2.0 * w * c + 2.0 * np.cross(q_vec, c, axis=-1)


def world_to_body(quat: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate a vector from world to body frame: v_body = R(q)^T @ v_world.

    Inverse of `body_to_world`. quat is (..., 4) in (w, x, y, z), v_world is (..., 3).
    """
    w = quat[..., 0:1]
    q_vec = quat[..., 1:4]
    c = np.cross(q_vec, v_world, axis=-1)
    return v_world - 2.0 * w * c + 2.0 * np.cross(q_vec, c, axis=-1)


def yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    """Yaw angle (rotation about world z) from quaternion (w, x, y, z).

    Args:
        quat: (..., 4) unit quaternion in (w, x, y, z) order.
    Returns:
        theta: (...,) yaw in radians, range (-pi, pi].
    """
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]
    return np.arctan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def q_to_rom(q: np.ndarray) -> np.ndarray:
    """Reduce full qpos to a reduced-order pose [px, py, theta] in the world frame.

    Args:
        q: (..., nq) with q[..., 0:3] = base position, q[..., 3:7] = quat (w, x, y, z).
    Returns:
        q_rom: (..., 3) of [px, py, theta].
    """
    px = q[..., 0]
    py = q[..., 1]
    theta = yaw_from_quat(q[..., 3:7])
    return np.stack([px, py, theta], axis=-1)


def v_to_rom(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reduce full qvel to a reduced-order twist [vx, vy, omega] in the body frame.

    MuJoCo's free-joint qvel stores world-frame linear and angular velocity;
    this rotates both into the base body frame using the base quaternion
    and returns the (x, y) components of the linear part and the z component
    of the angular part — matching what a body-frame velocity command (vx, vy, wz)
    is expected to drive.

    Args:
        q: (..., nq) full qpos; q[..., 3:7] is the base quaternion.
        v: (..., nv) full qvel; v[..., 0:3] world-frame linear, v[..., 3:6] world-frame angular.
    Returns:
        v_rom: (..., 3) of [vx_body, vy_body, omega_z_body].
    """
    quat = q[..., 3:7]
    v_lin_body = world_to_body(quat, v[..., 0:3])   # (..., 3)
    v_ang_body = world_to_body(quat, v[..., 3:6])   # (..., 3)
    return np.stack([v_lin_body[..., 0],
                     v_lin_body[..., 1],
                     v_ang_body[..., 2]], axis=-1)
