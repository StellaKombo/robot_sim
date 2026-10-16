##
#
#  KCF origin controller -> body-frame [vx, vy, wz].
#  Run simulation.py separately, then this file (CPU defaults to one Torch thread).
#  --max-input-rate and --input-delta-weight control command smoothing.
#
##

# standard imports
from time import perf_counter
import argparse
import warnings

import numpy as np

# custom imports (control_base puts the repo root on sys.path for the imports after it)
from control_base import CommandSource, RobotState, run
from policy.config import G1Config
from utils.math_utils import gravity_from_quat, q_to_rom, v_to_rom, world_to_body

# import KCF
import sys
from pathlib import Path
KCF_ROOT = Path(__file__).resolve().parents[3] / "kcf"
sys.path.insert(0, str(KCF_ROOT))

from mpc import create_mpc_controller, quadratic_cost
from config.mpc_config import get_default_mpc_config
from scipy.optimize import OptimizeResult, minimize
import torch

############################################################################
# KCF CLOSED-LOOP STATE
############################################################################

def g1_closed_loop_state(state: RobotState, gait_period: float, theta_prev: float | None = None) -> np.ndarray:
    """84-D observation in the saved KCF checkpoint's training convention.

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
    # Preserve the exact training features, even though qvel angular velocity is
    # already body-frame in MuJoCo. Changing this alone would mismatch the weights.
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
    """KCF MPC holding the origin, returning body-frame [vx, vy, wz]."""

    def __init__(self, cmd_scale: np.ndarray, mpc_config=None,
                 max_input_rate: float = 1.5, input_delta_weight: float = 1.0,
                 verbose: bool = False):
        
        super().__init__(cmd_scale)
        self.cfg = G1Config()

        # MPC controller
        settings = get_default_mpc_config()
        settings["device"] = "cpu"
        # Prioritize returning to the origin while still penalizing speed at the target.
        settings["state_weights"] = (10.0, 10.0, 2.0, 1.0, 1.0, 1.0)
        if mpc_config is not None:
            settings.update(mpc_config)
        self.mpc = create_mpc_controller(settings)
        if (self.mpc.model.config.state_dim != 84 or self.mpc.model.config.L_embed != 0
                or self.mpc.input_dim != 3
                or tuple(self.mpc.tracked_state_indices.cpu().tolist()) != (0, 1, 2, 3, 4, 5)):
            raise ValueError("G1 controller requires an 84-input-state model tracking ROM6 with three controls")
        if np.any(self.mpc.lower_bounds < -self.cmd_scale) or np.any(self.mpc.upper_bounds > self.cmd_scale):
            raise ValueError("MPC input bounds exceed the simulator command limits")
        self.N = self.mpc.horizon
        self.dt = self.cfg.control_dt   # [s] MPC step, the same as the KCF model_dt

        # KCF closed-loop state: gait period from the policy config, and last tick's unwrapped yaw
        self.gait_period = self.cfg.gait_period
        self.theta_prev = None
        self.previous_command = np.zeros(3, dtype=np.float32)
        self.last_state_time = None
        self.last_observation = None
        self.start_time_seen = None
        self.solve_time = 0.0
        self.verbose = verbose
        self.max_input_rate = float(max_input_rate)
        self.input_delta_weight = float(input_delta_weight)
        if not np.isfinite(self.max_input_rate) or self.max_input_rate <= 0:
            raise ValueError("max_input_rate must be finite and positive")
        if not np.isfinite(self.input_delta_weight) or self.input_delta_weight < 0:
            raise ValueError("input_delta_weight must be finite and non-negative")

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

    def reset(self):
        self.theta_prev = None
        self.previous_command[:] = 0.0
        self.last_state_time = None
        self.last_observation = None
        self.start_time_seen = self.t_start
        self.mpc.reset()

    def _fallback(self, reason):
        change = self.max_input_rate * self.dt
        command = np.sign(self.previous_command) * np.maximum(np.abs(self.previous_command) - change, 0.0)
        command = np.clip(command, self.mpc.lower_bounds, self.mpc.upper_bounds)
        self.mpc.reset()
        self.mpc_result = OptimizeResult(success=False, used_fallback=True, message=str(reason))
        return command.astype(np.float32)

    def _solve(self, observation, reference):
        """Use the saved KCF dynamics with local smoothing and solver-result checks."""
        mpc = self.mpc
        x = torch.as_tensor(observation, device=mpc.device, dtype=mpc.dtype)
        ref = torch.as_tensor(reference, device=mpc.device, dtype=mpc.dtype)
        previous = torch.as_tensor(self.previous_command, device=mpc.device, dtype=mpc.dtype)
        if x.shape != (84,) or ref.shape != (self.N, 6):
            raise ValueError("Unexpected MPC state or reference shape")
        if not torch.isfinite(x).all() or not torch.isfinite(ref).all():
            raise FloatingPointError("Non-finite MPC observation or reference")
        with torch.no_grad():
            lifted = mpc.model.lift(x[None]).squeeze(0)

        def objective(flat_controls):
            controls = torch.tensor(flat_controls.reshape(self.N, 3), device=mpc.device,
                                    dtype=mpc.dtype, requires_grad=True)
            cost = quadratic_cost(mpc.model, lifted, controls, ref,
                                  mpc.tracked_state_indices, mpc.state_weights, mpc.input_weight)
            changes = torch.cat((controls[:1] - previous, controls[1:] - controls[:-1]))
            cost = cost + self.input_delta_weight * changes.square().sum()
            gradient, = torch.autograd.grad(cost, controls)
            if not torch.isfinite(cost) or not torch.isfinite(gradient).all():
                raise FloatingPointError("Non-finite MPC cost or gradient")
            return float(cost.detach().cpu()), gradient.detach().cpu().numpy().astype(np.float64).ravel()

        lower = np.tile(mpc.lower_bounds, self.N)
        upper = np.tile(mpc.upper_bounds, self.N)
        change = self.max_input_rate * self.dt
        lower[:3] = np.maximum(lower[:3], self.previous_command - change)
        upper[:3] = np.minimum(upper[:3], self.previous_command + change)
        guess = np.clip(mpc._initial_guess().ravel(), lower, upper)
        initial_cost, _ = objective(guess)
        result = minimize(objective, guess, method="L-BFGS-B", jac=True,
                          bounds=list(zip(lower, upper)), options={
                              "maxiter": mpc.max_iterations,
                              "ftol": mpc.function_tolerance,
                              "gtol": mpc.gradient_tolerance})
        candidate = np.asarray(result.x, dtype=np.float64)
        if not result.success:
            return self._fallback(result.message)
        if (candidate.shape != (self.N * 3,) or not np.isfinite(candidate).all()
                or np.any(candidate < lower - 1e-7) or np.any(candidate > upper + 1e-7)):
            return self._fallback("Optimizer returned invalid or infeasible controls")
        candidate = np.clip(candidate, lower, upper)
        cost, _ = objective(candidate)
        if cost > initial_cost + 1e-6 * max(1.0, abs(initial_cost)):
            return self._fallback("Optimizer increased the objective")
        solution = candidate.reshape(self.N, 3)
        mpc.previous_solution = solution.copy()
        result.used_fallback = False
        self.mpc_result = result
        return solution[0].astype(np.float32)

    def get(self, state: RobotState) -> np.ndarray:
        if (self.start_time_seen != self.t_start
                or (self.last_state_time is not None and state.t < self.last_state_time)):
            self.reset()
        x_kcf = g1_closed_loop_state(state, self.gait_period, self.theta_prev)
        if self.last_state_time == state.t:
            if np.array_equal(x_kcf, self.last_observation):
                return self.previous_command.copy()
            self.mpc.reset()
        if np.isfinite(x_kcf[2]):
            self.theta_prev = float(x_kcf[2])
        t = state.t - self.t_start
        q_ref, v_ref = self.reference(t + self.dt * np.arange(1, self.N + 1))
        reference = np.concatenate([q_ref, v_ref], axis=-1).astype(np.float32)
        self.ref = q_ref[0, :2]
        began = perf_counter()
        try:
            command = self._solve(x_kcf, reference)
        except (FloatingPointError, RuntimeError) as error:
            command = self._fallback(error)
        self.solve_time = perf_counter() - began
        if self.mpc_result.used_fallback:
            warnings.warn(f"MPC fallback at sim t={state.t:.3f}: {self.mpc_result.message}")
        self.previous_command = command.copy()
        self.last_state_time = state.t
        self.last_observation = x_kcf.copy()
        if self.verbose:
            status = "fallback" if self.mpc_result.used_fallback else "converged"
            print(f"t={state.t:.3f}s solve={self.solve_time:.3f}s {status} u={command}")
        return command.copy()

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



def main():
    parser = argparse.ArgumentParser(description="G1 KCF controller holding the origin")
    parser.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    parser.add_argument("--threads", type=int, default=1, help="CPU PyTorch threads (default: 1)")
    parser.add_argument("--results-dir", type=Path, default=None, help="override the saved model directory")
    parser.add_argument("--max-input-rate", type=float, default=1.5,
                        help="command slew rate in units/s of model time (default: 1.5)")
    parser.add_argument("--input-delta-weight", type=float, default=1.0, help="command-change penalty")
    parser.add_argument("--position-weight", type=float, default=10.0, help="weight on x/y position errors")
    parser.add_argument("--verbose", action="store_true", help="show solve times, commands, and status")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    config = {"device": args.device,
              "state_weights": (args.position_weight, args.position_weight, 2.0, 1.0, 1.0, 1.0)}
    if args.results_dir is not None:
        config["results_dir"] = str(args.results_dir.resolve())
    cfg = G1Config()
    source = CustomCommand(cfg.cmd_scale, config, args.max_input_rate, args.input_delta_weight, args.verbose)
    try:
        run(source, cfg)
    finally:
        source.close()


if __name__ == "__main__":
    main()
