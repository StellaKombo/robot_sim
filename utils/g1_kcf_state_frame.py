"""Shared schema adapters for legacy 84-D and new 85-D sine/cosine G1 states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.relative_frame import RelativeFrame

__all__ = ["G1_KCF_STATE_DIM", "G1_RELATIVE_WINDOW_SCHEMA", "G1KCFRelativeFrame",
           "G1_SINCOS_STATE_DIM", "G1_RELATIVE_SINCOS_SCHEMA", "G1KCFSinCosFrame"]

G1_KCF_STATE_DIM = 84
G1_RELATIVE_WINDOW_SCHEMA = "g1_kcf_relative84_window_v1"
G1_SINCOS_STATE_DIM = 85
G1_RELATIVE_SINCOS_SCHEMA = "g1_kcf_relative85_sincos_window_v1"
_POSE_SLICE = slice(0, 3)
_UNCHANGED_SLICE = slice(3, G1_KCF_STATE_DIM)
_FRAME_TOLERANCE = 1e-5


def _states(values, name: str, state_dim=G1_KCF_STATE_DIM) -> np.ndarray:
    """Validate finite floating-point KCF values of the declared width."""
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != state_dim:
        raise ValueError(
            f"{name} must have shape (..., {state_dim})"
        )
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain numeric values")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _planar_poses(values, name: str) -> np.ndarray:
    """Validate one or more ``[px, py, yaw_unwrapped]`` poses."""
    array = np.asarray(values)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise TypeError(f"{name} must contain numeric values")
    if not np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class G1KCFRelativeFrame:
    """Apply one fixed ground/yaw frame to the existing 84-D KCF schema.

    Only ``[px, py, yaw_unwrapped]`` is transformed.  Channels ``3:84`` are
    copied bit-for-bit because the runtime state defines them in body-fixed or
    otherwise frame-invariant coordinates.

    ``yaw_origin_world`` is stored separately from the frame quaternion so an
    unwrapped heading can retain its continuous turn count.  The quaternion
    still supplies the planar rotation used for position transformations.
    """

    frame: RelativeFrame
    yaw_origin_world: float

    def __post_init__(self) -> None:
        yaw_origin = float(self.yaw_origin_world)
        if not np.isfinite(yaw_origin):
            raise ValueError("yaw_origin_world must be finite")
        if abs(float(self.frame.origin_world[2])) > _FRAME_TOLERANCE:
            raise ValueError("the KCF frame origin must be ground-anchored at z=0")
        if np.max(np.abs(self.frame.orientation_world[1:3])) > _FRAME_TOLERANCE:
            raise ValueError("the KCF frame must contain yaw rotation only")
        heading_error = np.arctan2(
            np.sin(yaw_origin - self.frame.heading_world),
            np.cos(yaw_origin - self.frame.heading_world),
        )
        if abs(float(heading_error)) > _FRAME_TOLERANCE:
            raise ValueError(
                "yaw_origin_world must match the frame heading modulo 2*pi"
            )
        object.__setattr__(self, "yaw_origin_world", yaw_origin)

    @classmethod
    def from_anchor(
        cls,
        anchor_state_world,
        orientation_world,
    ) -> "G1KCFRelativeFrame":
        """Create a fixed frame from one 84-D state and its base quaternion."""
        anchor = _states(anchor_state_world, "anchor_state_world")
        if anchor.shape != (G1_KCF_STATE_DIM,):
            raise ValueError("anchor_state_world must have shape (84,)")
        ground_origin = np.array(
            [anchor[0], anchor[1], 0.0], dtype=anchor.dtype
        )
        frame = RelativeFrame.from_pose(
            ground_origin, orientation_world, yaw_only=True
        )
        return cls(frame=frame, yaw_origin_world=float(anchor[2]))

    @classmethod
    def from_planar_state(cls, anchor_state_world) -> "G1KCFRelativeFrame":
        """Create a yaw-aligned frame when only the 84-D KCF state is known.

        Live MuJoCo adapters should prefer :meth:`from_anchor` with the base
        quaternion.  MPC already receives the continuous planar heading in the
        KCF state, so reconstructing its yaw-only quaternion here avoids a
        second coordinate-frame implementation in either numerical backend.
        """
        anchor = _states(anchor_state_world, "anchor_state_world")
        if anchor.shape != (G1_KCF_STATE_DIM,):
            raise ValueError("anchor_state_world must have shape (84,)")
        half_yaw = 0.5 * float(anchor[2])
        orientation = np.array(
            [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
            dtype=np.result_type(anchor.dtype, np.float32),
        )
        return cls.from_anchor(anchor, orientation)

    def planar_pose_to_local(self, pose_world) -> np.ndarray:
        """Transform ``[..., px, py, yaw_unwrapped]`` into this fixed frame."""
        pose = _planar_poses(pose_world, "pose_world")
        result = np.array(pose, copy=True)
        position = np.zeros(
            (*pose.shape[:-1], 3),
            dtype=np.result_type(pose.dtype, self.frame.origin_world.dtype),
        )
        position[..., :2] = pose[..., :2]
        result[..., :2] = self.frame.position_to_local(position)[..., :2]
        result[..., 2] = pose[..., 2] - self.yaw_origin_world
        return result

    def planar_pose_to_world(self, pose_local) -> np.ndarray:
        """Invert :meth:`planar_pose_to_local` for one or more poses."""
        pose = _planar_poses(pose_local, "pose_local")
        result = np.array(pose, copy=True)
        position = np.zeros(
            (*pose.shape[:-1], 3),
            dtype=np.result_type(pose.dtype, self.frame.origin_world.dtype),
        )
        position[..., :2] = pose[..., :2]
        result[..., :2] = self.frame.position_to_world(position)[..., :2]
        result[..., 2] = pose[..., 2] + self.yaw_origin_world
        return result

    def state_to_local(self, state_world) -> np.ndarray:
        """Transform world KCF states while preserving their 84-D layout."""
        state = _states(state_world, "state_world")
        result = np.array(state, copy=True)
        result[..., :3] = self.planar_pose_to_local(state[..., :3])
        return result

    def state_to_world(self, state_local) -> np.ndarray:
        """Invert :meth:`state_to_local` in the unchanged 84-D layout."""
        state = _states(state_local, "state_local")
        result = np.array(state, copy=True)
        result[..., :3] = self.planar_pose_to_world(state[..., :3])
        return result

    @property
    def transformed_indices(self) -> tuple[int, int, int]:
        """Indices changed by the adapter, useful for schema audits."""
        return tuple(range(_POSE_SLICE.start, _POSE_SLICE.stop))

    @property
    def unchanged_slice(self) -> slice:
        """Contiguous body/intrinsic portion copied without transformation."""
        return _UNCHANGED_SLICE


@dataclass(frozen=True)
class G1KCFSinCosFrame:
    """85-D schema adapter; all coordinate mathematics belongs to RelativeFrame.

    Pose is [px, py, sin(heading), cos(heading)]. Body/intrinsic channels 4:85
    are copied unchanged. The legacy 84-D adapter above remains separate.
    """

    frame: RelativeFrame

    def __post_init__(self):
        if (abs(float(self.frame.origin_world[2])) > _FRAME_TOLERANCE
                or np.max(np.abs(self.frame.orientation_world[1:3])) > _FRAME_TOLERANCE):
            raise ValueError("the KCF frame must be ground-anchored and contain yaw only")

    @classmethod
    def from_state(cls, anchor_state_world):
        anchor = _states(anchor_state_world, "anchor_state_world", G1_SINCOS_STATE_DIM)
        if anchor.shape != (G1_SINCOS_STATE_DIM,):
            raise ValueError("anchor_state_world must have shape (85,)")
        if not np.isclose(np.linalg.norm(anchor[2:4]), 1., atol=1e-5):
            raise ValueError("heading sine/cosine must have unit norm")
        half_yaw = .5 * np.arctan2(anchor[2], anchor[3])
        return cls(RelativeFrame.from_pose(
            [anchor[0], anchor[1], 0.],
            [np.cos(half_yaw), 0., 0., np.sin(half_yaw)], yaw_only=True,
        ))

    def _pose(self, pose, *, inverse=False):
        pose = _states(pose, "pose", 4)
        if np.any(np.linalg.norm(pose[..., 2:4], axis=-1) < 1e-12):
            raise ValueError("heading sine/cosine must have nonzero magnitude")
        precise = pose.astype(np.float64, copy=False)
        position = np.zeros((*pose.shape[:-1], 3), dtype=np.float64)
        position[..., :2] = precise[..., :2]
        if inverse:
            position = self.frame.position_to_world(position)
            heading = self.frame.heading_sincos_to_world(precise[..., 2:4])
            heading = np.stack((np.sin(heading), np.cos(heading)), axis=-1)
        else:
            position = self.frame.position_to_local(position)
            heading = self.frame.heading_to_local_sincos(np.arctan2(precise[..., 2], precise[..., 3]))
        return np.concatenate((position[..., :2], heading), axis=-1).astype(pose.dtype)

    def pose_to_local(self, pose_world):
        return self._pose(pose_world)

    def state_to_local(self, state_world):
        result = _states(state_world, "state_world", G1_SINCOS_STATE_DIM).copy()
        result[..., :4] = self.pose_to_local(result[..., :4])
        return result

    def state_to_world(self, state_local):
        result = _states(state_local, "state_local", G1_SINCOS_STATE_DIM).copy()
        result[..., :4] = self._pose(result[..., :4], inverse=True)
        return result
