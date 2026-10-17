##
#
#  Custom controller: KCF MPC -> body-frame [vx, vy, wz].
#
##

# standard imports
from pathlib import Path
import sys
from time import perf_counter

# third-party imports
import numpy as np
import torch
from scipy.optimize import OptimizeResult  # Failure metadata only, not another optimizer.

# control_base puts the robot_sim root on sys.path
from control_base import ROOT_DIR, CommandSource, RobotState, main
from policy.config import G1Config
from utils.math_utils import gravity_from_quat, q_to_rom, v_to_rom, world_to_body

KCF_ROOT = Path(ROOT_DIR) / "KCF_MPC"
sys.path.insert(0, str(KCF_ROOT))

from config.mpc_config import get_default_mpc_config
from mpc import create_mpc_controller


############################################################################
# KCF CLOSED-LOOP STATE
############################################################################

def g1_closed_loop_state(state: RobotState, gait_period: float,
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
    gravity = gravity_from_quat(quat)
    linear_body = world_to_body(quat, v[:3])
    # Preserve the checkpoint's velocity convention, including this rotation.
    angular_body = world_to_body(quat, v[3:6])
    remaining_base_velocity = np.array(
        [linear_body[2], angular_body[0], angular_body[1]], dtype=np.float32
    )
    phase = np.remainder(state.t, gait_period) / gait_period
    gait = np.array([np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32)
    return np.concatenate([planar_pose, planar_velocity, q[2:3], gravity, q[7:],
                           remaining_base_velocity, v[6:], gait, state.prev_action.astype(np.float32)])


############################################################################
# CONTROLLER
############################################################################


class CustomCommand(CommandSource):
    """Track reference() using the existing KCF MPC; hold the origin for now.

    Keep the baseline's reference and Lissajous functions. The optimizer,
    rollout, cost and existing input bounds belong to KCF_MPC/mpc.py.
    """

    def __init__(self, cmd_scale: np.ndarray, *, quiet: bool = False, verbose: bool = False):
        super().__init__(cmd_scale)
        self.cfg = G1Config()
        settings = get_default_mpc_config()
        threads = int(settings["num_threads"])
        if threads < 1:
            raise ValueError("MPC num_threads must be positive")
        torch.set_num_threads(threads)
        self.mpc = create_mpc_controller(settings)

        dtypes = {"float32": torch.float32, "float64": torch.float64}
        dtype = dtypes[settings["dtype"]]
        self.mpc.model.to(dtype=dtype)
        self.mpc.dtype = dtype
        self.mpc.state_weights = self.mpc.state_weights.to(dtype=dtype)
        if (self.mpc.model.config.state_dim != 84 or self.mpc.model.config.L_embed != 0
                or self.mpc.input_dim != 3
                or tuple(self.mpc.tracked_state_indices.cpu().tolist()) != (0, 1, 2, 3, 4, 5)):
            raise ValueError("G1 requires the 84-state checkpoint tracking ROM6 with three commands")
        if np.any(self.mpc.lower_bounds < -self.cmd_scale) or np.any(self.mpc.upper_bounds > self.cmd_scale):
            raise ValueError("MPC input bounds exceed the simulator's command limits")
        self.reference_state = np.asarray(settings["reference"], dtype=np.float64)
        if self.reference_state.shape != (6,) or not np.isfinite(self.reference_state).all():
            raise ValueError("MPC reference must contain six finite ROM values")
        self.N = self.mpc.horizon
        self.dt = self.cfg.control_dt
        self.gait_period = self.cfg.gait_period
        # Retain the baseline's optional trajectory; it is not active for origin holding.
        trajectory = settings["lissajous_reference"]
        self.half_width_ref = float(trajectory["half_width"])
        self.half_height_ref = float(trajectory["half_height"])
        self.omega_ref = float(trajectory["omega"])
        self.quiet, self.verbose = quiet, verbose
        self.solve_count = self.failure_count = self.deadline_miss_count = 0
        self.solve_time = 0.0
        self.reset()
        if not self.quiet:
            print(f"Using {KCF_ROOT / 'mpc.py'} (L-BFGS-B); "
                  f"settings from {KCF_ROOT / 'config' / 'mpc_config.py'}", flush=True)

    def reference(self, t):
        """Reference pose and twist at scalar/array controller time t; hold zero for now.

        Returns (q_ref, v_ref), each (..., 3): world pose [px, py, theta]
        and world twist [vx, vy, wz]. Values are loaded from the MPC config.
        To track the baseline's curve later, return self.lissajous_reference(t).
        """
        shape = np.shape(t) + (3,)
        return (np.broadcast_to(self.reference_state[:3], shape).copy(),
                np.broadcast_to(self.reference_state[3:], shape).copy())

    def lissajous_reference(self, t):
        """Baseline Lissajous pose/twist in the world frame, with tangent heading."""
        a, b, w = self.half_width_ref, self.half_height_ref, self.omega_ref
        t = np.asarray(t, dtype=np.float64)
        px, py = a * np.sin(w * t), b * np.sin(2 * w * t)
        vx, vy = a * w * np.cos(w * t), 2 * b * w * np.cos(2 * w * t)
        ax, ay = -a * w**2 * np.sin(w * t), -4 * b * w**2 * np.sin(2 * w * t)
        theta = np.arctan2(vy, vx)
        wz = (vx * ay - vy * ax) / (vx**2 + vy**2)
        return np.stack([px, py, theta], axis=-1), np.stack([vx, vy, wz], axis=-1)

    def reset(self):
        self.theta_prev = None
        self.last_state_time = None
        self.last_observation = None
        self.previous_command = np.zeros(3, dtype=np.float32)
        self.start_time_seen = self.t_start
        self.mpc.reset()

    def get(self, state: RobotState) -> np.ndarray:
        if (self.start_time_seen != self.t_start
                or (self.last_state_time is not None and state.t < self.last_state_time)):
            self.reset()
        # Controller clock and 84-D observation follow the original baseline.
        t = state.t - self.t_start
        x_kcf = g1_closed_loop_state(state, self.gait_period, self.theta_prev)
        if self.last_state_time == state.t:
            if np.array_equal(x_kcf, self.last_observation):
                return self.previous_command.copy()
            self.mpc.reset()
        interval = None if self.last_state_time is None else state.t - self.last_state_time
        if np.isfinite(x_kcf[2]):
            self.theta_prev = float(x_kcf[2])
        # The MPC predicts x[k+1] ... x[k+N], not the current state x[k].
        t_horizon = t + self.dt * np.arange(1, self.N + 1)
        q_ref, v_ref = self.reference(t_horizon)
        # Reference functions return world velocity; KCF's ROM twist is body-frame.
        cos_yaw, sin_yaw = np.cos(q_ref[:, 2]), np.sin(q_ref[:, 2])
        v_ref_body = np.stack([cos_yaw * v_ref[:, 0] + sin_yaw * v_ref[:, 1],
                               -sin_yaw * v_ref[:, 0] + cos_yaw * v_ref[:, 1],
                               v_ref[:, 2]], axis=-1)
        x_ref = np.concatenate([q_ref, v_ref_body], axis=-1)  # (N, 6), as solve expects
        self.ref = q_ref[0, :2].copy()
        began = perf_counter()
        try:
            # The sole optimization call: KCF's original cost and L-BFGS-B solver.
            u_seq, result = self.mpc.solve(x_kcf, x_ref)
            result.used_fallback = not result.success
            if not result.success:
                command = np.zeros(3, dtype=np.float32)
                self.mpc.reset()
            else:
                command = np.asarray(u_seq[0], dtype=np.float32)
                if (not np.isfinite(command).all()
                        or np.any(command < self.mpc.lower_bounds - 1e-7)
                        or np.any(command > self.mpc.upper_bounds + 1e-7)):
                    raise RuntimeError("KCF MPC returned an invalid or out-of-bounds command")
        except (FloatingPointError, RuntimeError, ValueError) as error:
            command = np.zeros(3, dtype=np.float32)
            result = OptimizeResult(success=False, used_fallback=True, message=str(error))
            self.mpc.reset()
        self.solve_time = perf_counter() - began
        self.mpc_result = result
        self.solve_count += 1
        self.failure_count += int(not result.success)
        self.deadline_miss_count += int(self.solve_time > self.dt)
        self.last_state_time, self.last_observation = state.t, x_kcf.copy()
        self.previous_command = command.copy()
        if not self.quiet or self.verbose or not result.success:
            status = "converged" if result.success else "fallback"
            state_dt = "n/a" if interval is None else f"{interval:.3f}s"
            print(f"[MPC L-BFGS-B] t={state.t:.3f}s state_dt={state_dt} "
                  f"compute={1000 * self.solve_time:.1f}ms iterations={result.get('nit', 'n/a')} "
                  f"evaluations={result.get('nfev', 'n/a')} status={status} "
                  f"failures={self.failure_count}/{self.solve_count} "
                  f"overruns={self.deadline_miss_count}/{self.solve_count} "
                  f"position_error={np.linalg.norm(state.q[:2] - self.ref):.3f}m",
                  flush=True)
            if self.verbose or not result.success:
                print(f"      command={command} reason={result.message}", flush=True)
        return command.copy()


if __name__ == "__main__":
    main(CustomCommand, "G1 custom controller: KCF L-BFGS-B MPC")
