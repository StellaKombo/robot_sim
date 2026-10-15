##
#
#  Custom controller: your own logic -> body-frame [vx, vy, wz].
#
##

# standard imports
import numpy as np

# custom imports
from control_base import CommandSource, RobotState, main


class CustomCommand(CommandSource):
    """Lissajous-curve tracking: PD on planar position and heading, returned as body-frame [vx, vy, wz].

    Reference p(t) = [a sin(w t), b sin(2 w t)] crosses itself at (0, 0) at t = 0, and the
    reference heading follows its tangent. Called every control_dt; the sim clips to cmd_scale.
    """

    def __init__(self, cmd_scale: np.ndarray):
        super().__init__(cmd_scale)

        # Lissajous curve size and rate: one loop every 2 pi / w seconds
        self.half_width_ref = 1.5    # [m] a: reaches x = +-a
        self.half_height_ref = 0.75  # [m] b: each lobe reaches y = +-b
        self.omega_ref = 0.25         # [rad/s] w

        # PD gains on planar position and on heading
        self.kp_pos, self.kd_pos = 1.0, 0.2
        self.kp_yaw, self.kd_yaw = 2.0, 0.1

    # feedback control based
    def get(self, state: RobotState) -> np.ndarray:

        # controller time: zero at the first state from the sim
        t = state.t - self.t_start

        # unpack full state
        qpos_base = state.q[:7]
        qvel_base = state.v[:6]
        # qpos_joints = state.q[7:]  # not used  
        # qvel_joints = state.v[6:]  # not used

        # unpack planar dynamics state
        px, py = qpos_base[:2]    # planar positions
        vx, vy = qvel_base[:2]    # planar velocities
        wz = qvel_base[5]         # base frame ang vel
        qw, qx, qy, qz = qpos_base[3:7] # base quat
        theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))  # yaw [rad]

        # reference position, velocity, and acceleration along the Lissajous curve
        a, b, w = self.half_width_ref, self.half_height_ref, self.omega_ref
        px_ref, py_ref = a * np.sin(w * t), b * np.sin(2 * w * t)
        vx_ref, vy_ref = a * w * np.cos(w * t), 2 * b * w * np.cos(2 * w * t)
        ax_ref, ay_ref = -a * w**2 * np.sin(w * t), -4 * b * w**2 * np.sin(2 * w * t)
        self.ref = [px_ref, py_ref]   # drawn as a ball in the sim

        # heading along the tangent, and its rate from the curvature (speed is never zero here)
        theta_ref = np.arctan2(vy_ref, vx_ref)
        wz_ref = (vx_ref * ay_ref - vy_ref * ax_ref) / (vx_ref**2 + vy_ref**2)

        # PD on planar position, in the world frame, on top of the reference velocity
        vx_cmd = vx_ref + self.kp_pos * (px_ref - px) + self.kd_pos * (vx_ref - vx)
        vy_cmd = vy_ref + self.kp_pos * (py_ref - py) + self.kd_pos * (vy_ref - vy)

        # PD on heading, with the angle error wrapped to [-pi, pi]
        yaw_err = np.arctan2(np.sin(theta_ref - theta), np.cos(theta_ref - theta))
        wz_cmd = wz_ref + self.kp_yaw * yaw_err + self.kd_yaw * (wz_ref - wz)

        # rotate the planar velocity command into the body frame
        vx_body =  np.cos(theta) * vx_cmd + np.sin(theta) * vy_cmd
        vy_body = -np.sin(theta) * vx_cmd + np.cos(theta) * vy_cmd

        return np.array([vx_body, vy_body, wz_cmd], dtype=np.float32)


if __name__ == "__main__":
    main(CustomCommand, "G1 custom controller")
