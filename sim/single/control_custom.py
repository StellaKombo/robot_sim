##
#
#  Custom controller: your own logic -> body-frame [vx, vy, wz].
#
##

# standard imports
import numpy as np

# custom imports (control_base puts the repo root on sys.path for the imports after it)
from control_base import CommandSource, RobotState, main
from policy.config import G1Config
from utils.math_utils import gravity_from_quat, q_to_rom, v_to_rom, world_to_body

# import KCF
import sys
from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent.parent / "kcf"))
# from kcf import G1KCF, G1KCFConfig  

KCF_ROOT = Path(__file__).resolve().parents[3] / "KCF_MPC"
sys.path.insert(0, str(KCF_ROOT))

from mpc import create_mpc_controller

############################################################################
# KCF CLOSED-LOOP STATE
############################################################################

def g1_closed_loop_state(state: RobotState, gait_period: float, theta_prev: float | None = None) -> np.ndarray:
    """84-D Markov state of the full-order KCF model at one control tick, as float32.

    Single-tick version of the batched _g1_build_closed_loop_state, with the same layout:
    [px, py, theta] (3), [vx, vy, wz] (3), pz (1), gravity (3), joint q (23), [vz, wx, wy] (3),
    joint v (23), gait [sin, cos] (2), previous action (23, state.prev_action). The batched version
    unwraps theta over time, so pass the previous tick's theta (x[2]) as theta_prev, or None at first.
    """
    # float32, like the logged datasets the batched version reads
    q = state.q.astype(np.float32)
    v = state.v.astype(np.float32)
    quat = q[3:7]

    # planar pose, with yaw unwrapped against the previous tick, and planar twist
    planar_pose = q_to_rom(q).astype(np.float32, copy=False)
    if theta_prev is not None:
        planar_pose[2] = np.unwrap(np.array([theta_prev, planar_pose[2]], dtype=np.float32))[1]
    planar_velocity = v_to_rom(q, v).astype(np.float32, copy=False)

    # projected gravity, and the base velocities the planar twist leaves out; same frame
    # handling as the batched version (and v_to_rom), so the features match the training data
    gravity = gravity_from_quat(quat)
    linear_body = world_to_body(quat, v[0:3])
    angular_body = world_to_body(quat, v[3:6])
    remaining_base_velocity = np.array([linear_body[2], angular_body[0], angular_body[1]], dtype=np.float32)

    # gait clock from sim time
    phase = np.remainder(state.t, gait_period) / gait_period
    gait_phase = np.array([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float32)

    return np.concatenate([planar_pose, planar_velocity, q[2:3], gravity, q[7:],
                           remaining_base_velocity, v[6:], gait_phase,
                           state.prev_action.astype(np.float32)])


############################################################################
# CONTROLLER
############################################################################

class CustomCommand(CommandSource):
    """Lissajous-curve tracking: PD on planar position and heading, returned as body-frame [vx, vy, wz].

    Reference p(t) = [a sin(w t), b sin(2 w t)] crosses itself at (0, 0) at t = 0, and the
    reference heading follows its tangent. Called every control_dt; the sim clips to cmd_scale.
    """

    def __init__(self, cmd_scale: np.ndarray):
        
        super().__init__(cmd_scale)
        self.cfg = G1Config()

        # MPC controller
        self.mpc = create_mpc_controller()
        self.N = self.mpc.horizon
        self.dt = self.cfg.control_dt   # [s] MPC step, the same as the KCF model_dt

        # KCF closed-loop state: gait period from the policy config, and last tick's unwrapped yaw
        self.gait_period = self.cfg.gait_period
        self.theta_prev = None

        # Lissajous curve size and rate: one loop every 2 pi / w seconds
        self.half_width_ref = 1.5    # [m] a: reaches x = +-a
        self.half_height_ref = 0.75  # [m] b: each lobe reaches y = +-b
        self.omega_ref = 0.25         # [rad/s] w

        # # PD gains on planar position and on heading
        # self.kp_pos, self.kd_pos = 1.0, 0.2
        # self.kp_yaw, self.kd_yaw = 2.0, 0.1

    def reference(self, t):
        """Reference at controller time t [s], a scalar or an array: zero for now, i.e. hold the origin.

        Returns (q_ref, v_ref), each (..., 3): pose [px, py, theta] and twist [vx, vy, wz].
        To track the Lissajous curve instead, return self.lissajous_reference(t).
        """
        zeros = np.zeros(np.shape(t) + (3,))
        return zeros, zeros.copy()

    def lissajous_reference(self, t):
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

    # feedback control based
    def get(self, state: RobotState) -> np.ndarray:

        # controller time: zero at the first state from the sim
        t = state.t - self.t_start

        # 84-D KCF closed-loop state to query the model with; yaw unwrapping restarts with each sim run
        theta_prev = None if state.t == self.t_start else self.theta_prev
        x_kcf = g1_closed_loop_state(state, self.gait_period, theta_prev) # 84-D
        self.theta_prev = x_kcf[2]

        # print(x_kcf.shape)

        # reference at the N states the MPC predicts, one dt apart: rows [px, py, theta, vx, vy, wz]
        t_horizon = t + self.dt * np.arange(1, self.N + 1)
        q_ref, v_ref = self.reference(t_horizon)                                # each (N, 3)
        x_ref = np.concatenate([q_ref, v_ref], axis=-1).T.astype(np.float32)   # (6, N); solve takes x_ref.T
        self.ref = x_ref[:2, 0]   # next reference point, drawn as a ball in the sim

        # receding horizon: solve over N steps, warm-started from the last solution (dropped at the
        # start of each sim run), and send the first input
        if state.t == self.t_start:
            self.mpc.reset()
        u_seq, self.mpc_result = self.mpc.solve(x_kcf, x_ref.T)   # (N, 3) body-frame [vx, vy, wz]

        # # unpack full state
        # qpos_base = state.q[:7]
        # qvel_base = state.v[:6]
        # # qpos_joints = state.q[7:]  # not used  
        # # qvel_joints = state.v[6:]  # not used

        # # unpack planar dynamics state
        # px, py = qpos_base[:2]    # planar positions
        # vx, vy = qvel_base[:2]    # planar velocities
        # wz = qvel_base[5]         # base frame ang vel
        # qw, qx, qy, qz = qpos_base[3:7] # base quat
        # theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))  # yaw [rad]

        # # reference pose and twist on the Lissajous curve
        # (px_ref, py_ref, theta_ref), (vx_ref, vy_ref, wz_ref) = self.reference(t)
        # self.ref = [px_ref, py_ref]   # drawn as a ball in the sim

        # # PD on planar position, in the world frame, on top of the reference velocity
        # vx_cmd = vx_ref + self.kp_pos * (px_ref - px) + self.kd_pos * (vx_ref - vx)
        # vy_cmd = vy_ref + self.kp_pos * (py_ref - py) + self.kd_pos * (vy_ref - vy)

        # # PD on heading, with the angle error wrapped to [-pi, pi]
        # yaw_err = np.arctan2(np.sin(theta_ref - theta), np.cos(theta_ref - theta))
        # wz_cmd = wz_ref + self.kp_yaw * yaw_err + self.kd_yaw * (wz_ref - wz)

        # # rotate the planar velocity command into the body frame
        # vx_body =  np.cos(theta) * vx_cmd + np.sin(theta) * vy_cmd
        # vy_body = -np.sin(theta) * vx_cmd + np.cos(theta) * vy_cmd

        # first input of the MPC solution
        return u_seq[0].astype(np.float32)

    

if __name__ == "__main__":
    main(CustomCommand, "G1 custom controller")
