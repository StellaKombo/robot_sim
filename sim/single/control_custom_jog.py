##
#
#  Custom controller for the 29-DoF JOG sim: PD on the single-integrator planar model, tracking
#  a Lissajous curve, returned as body-frame [vx, vy, wz]. Same law and gains as the PD in
#  control_custom_walk.py; the curve is scaled so a unicycle jog can follow it. Run the sim with
#  --raw_cmd so the twist is used as is (only clipped); the jog sim never commands vy.
#
##

# standard imports
import numpy as np

# custom imports (control_base puts the repo root on sys.path for the imports after it)
from control_base import CommandSource, RobotState, main

JOG_NU = 29   # joints of the jog robot


############################################################################
# CONTROLLER
############################################################################

class CustomJogCommand(CommandSource):
    """Lissajous-curve tracking: PD on planar position and heading, returned as body-frame [vx, vy, wz].

    Reference p(t) = [a sin(w t), b sin(2 w t)] crosses itself at (0, 0) at t = 0, and the
    reference heading follows its tangent.
    """

    def __init__(self, cmd_scale: np.ndarray):
        super().__init__(cmd_scale)
        self.warned = False
        print("NOTE: start the jog sim with --raw_cmd (python sim/single/simulation_jog.py --raw_cmd);\n"
              "      without it the sim reads these velocities as stick positions and distorts them.")

        # Lissajous sized for a unicycle jog: speed 0.60-1.27 m/s and |wz| <= 0.38 rad/s, inside the
        # forward band (0.5-1.5 m/s, |wz| <= 0.5) with room for PD; one 52 s loop over 15 x 7.5 m
        self.half_width_ref = 7.5    # [m] a: reaches x = +-a
        self.half_height_ref = 3.75  # [m] b: each lobe reaches y = +-b
        self.omega_ref = 0.12        # [rad/s] w

        # PD gains on planar position and on heading
        self.kp_pos, self.kd_pos = 1.0, 0.2
        self.kp_yaw, self.kd_yaw = 2.0, 0.1

    def reference(self, t):
        """Lissajous reference at controller time t [s], a scalar or an array, in the world frame.

        Returns (q_ref, v_ref), each (..., 3): pose [px, py, theta] and twist [vx, vy, wz], with
        theta along the curve's tangent, in (-pi, pi].
        """
        a, b, w = self.half_width_ref, self.half_height_ref, self.omega_ref
        t = np.asarray(t, dtype=np.float64)

        # position, velocity, and acceleration along the curve
        px, py = a * np.sin(w * t), b * np.sin(2 * w * t)
        vx, vy = a * w * np.cos(w * t), 2 * b * w * np.cos(2 * w * t)
        ax, ay = -a * w**2 * np.sin(w * t), -4 * b * w**2 * np.sin(2 * w * t)

        # heading along the tangent, and its rate from the curvature (speed is never zero here)
        theta = np.arctan2(vy, vx)
        wz = (vx * ay - vy * ax) / (vx**2 + vy**2)
        return np.stack([px, py, theta], axis=-1), np.stack([vx, vy, wz], axis=-1)

    def get(self, state: RobotState) -> np.ndarray:

        # jog only; a 23-joint state is the placeholder before connecting, or the walk sim
        if len(state.prev_action) != JOG_NU:
            if state.t > 0.0 and not self.warned:
                print(f"control_custom_jog.py needs simulation_jog.py, but the sim sends a "
                      f"{len(state.prev_action)}-joint state; sending zero.")
                self.warned = True
            return np.zeros(3, dtype=np.float32)

        # controller time: zero at the first state from the sim
        t = state.t - self.t_start

        # unpack full state
        qpos_base = state.q[:7]
        qvel_base = state.v[:6]

        # unpack planar dynamics state
        px, py = qpos_base[:2]    # planar positions
        vx, vy = qvel_base[:2]    # planar velocities
        wz = qvel_base[5]         # base frame ang vel
        qw, qx, qy, qz = qpos_base[3:7] # base quat
        theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))  # yaw [rad]

        # reference pose and twist on the Lissajous curve
        (px_ref, py_ref, theta_ref), (vx_ref, vy_ref, wz_ref) = self.reference(t)
        self.ref = [px_ref, py_ref]   # drawn as a ball in the sim

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
    main(CustomJogCommand, "G1 custom jog controller (PD, single integrator)")
