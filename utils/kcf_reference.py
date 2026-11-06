"""World-frame paths and tracked pose/body-twist/gravity targets.
Reference generation is independent of KCF weights and live MPC scheduling.
"""

import numpy as np


class PlanarReference:
    """Configured fixed or Lissajous path, preserving continuous tangent yaw."""

    def __init__(self, settings):
        self.state = np.asarray(settings["reference"], dtype=np.float64)
        if (self.state.shape != (9,)
                or not np.isfinite(self.state).all()):
            raise ValueError(
                "MPC reference must contain nine finite planar pose, twist, and gravity values"
            )
        self.mode = settings.get("reference_mode", "fixed")
        if self.mode not in ("fixed", "lissajous"):
            raise ValueError("MPC reference_mode must be 'fixed' or 'lissajous'")
        trajectory = settings["lissajous_reference"]
        try:
            raw_trajectory = (
                trajectory["half_width"], trajectory["half_height"], trajectory["omega"]
            )
            self.half_width, self.half_height, self.omega = map(
                float, raw_trajectory
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "lissajous_reference requires half_width, half_height, and omega"
            ) from error
        if (any(isinstance(value, bool) for value in raw_trajectory)
                or not np.isfinite(
                    [self.half_width, self.half_height, self.omega]
                ).all()
                or min(self.half_width, self.half_height, self.omega) <= 0.0):
            raise ValueError("Lissajous dimensions and omega must be finite and positive")

    def reference(self, t):
        """Reference pose and world twist at scalar/array controller time ``t``.

        Returns (q_ref, v_ref), each (..., 3): world pose [px, py, theta]
        and world twist [vx, vy, wz].
        """
        if self.mode == "lissajous":
            return self.lissajous_reference(t)
        shape = np.shape(t) + (3,)
        return (np.broadcast_to(self.state[:3], shape).copy(),
                np.broadcast_to(self.state[3:6], shape).copy())

    def lissajous_reference(self, t):
        """Baseline Lissajous pose/twist in the world frame, with tangent heading."""
        a, b, w = self.half_width, self.half_height, self.omega
        t = np.asarray(t, dtype=np.float64)
        px, py = a * np.sin(w * t), b * np.sin(2 * w * t)
        vx, vy = a * w * np.cos(w * t), 2 * b * w * np.cos(2 * w * t)
        ax, ay = -a * w**2 * np.sin(w * t), -4 * b * w**2 * np.sin(2 * w * t)
        theta = np.arctan2(vy, vx)
        # Follow one continuous tangent-heading branch. Without this correction,
        # atan2 injects a spurious 2*pi MPC error on the left side of the curve.
        phase = np.remainder(w * t, 2 * np.pi)
        reverse_branch = ((phase >= 3 * np.pi / 4)
                          & (phase <= 5 * np.pi / 4)
                          & (theta > 0.0))
        theta = np.where(reverse_branch, theta - 2 * np.pi, theta)
        wz = (vx * ay - vy * ax) / (vx**2 + vy**2)
        return np.stack([px, py, theta], axis=-1), np.stack([vx, vy, wz], axis=-1)

    def track(self, q_ref, v_ref):
        """Build MPC [world pose, body twist, gravity] references."""
        cos_yaw, sin_yaw = np.cos(q_ref[..., 2]), np.sin(q_ref[..., 2])
        v_ref_body = np.stack(
            [cos_yaw * v_ref[..., 0] + sin_yaw * v_ref[..., 1],
             -sin_yaw * v_ref[..., 0] + cos_yaw * v_ref[..., 1],
             v_ref[..., 2]],
            axis=-1,
        )
        gravity_ref = np.broadcast_to(self.state[6:9], q_ref.shape)
        return np.concatenate([q_ref, v_ref_body, gravity_ref], axis=-1)
