##
#
#  Parallel G1 simulation using MuJoCo Warp (GPU) + PyTorch policy
#
##

# standard imports
import argparse
import os
import sys
import time
import math
import hashlib
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass

# torch imports
import torch

# mujoco, warp imports
import mujoco
import mujoco_warp as mjwarp
import warp as wp

# custom imports
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)
from policy.actor import load_policy
from policy.config import G1Config
from utils.math_utils import get_gravity_orientation_batch, q_to_rom, v_to_rom


COMMAND_PROFILES = ("constant_zoh", "multi_step_zoh", "smooth", "multisine", "reversals")
PROFILE_WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)


def dataset_paths(save_dir, profile, batch_size, samples, zoh_steps, seed, number):
    """Keep full-order names discoverable by KCF_MPC/data_config.py."""
    suffix = f"data_{profile}_B{batch_size}_T{samples}_zoh{zoh_steps}_seed{seed}_{number:02d}.npz"
    return tuple(Path(save_dir) / f"{name}_{suffix}" for name in ("g1_23dof", "g1_rom"))


def make_command_schedule(batch_size, queries, hold_steps, control_dt, lower, upper,
                          rng, profile="mixed", weights=PROFILE_WEIGHTS):
    """Bounded body commands on the policy clock; independent of the legacy RNG."""
    weights = np.asarray(weights, dtype=float)
    if (weights.shape != (5,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.)):
        raise ValueError("five nonnegative profile weights must sum to one")
    if profile not in (*COMMAND_PROFILES, "mixed"):
        raise ValueError(f"unknown command profile: {profile}")
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all()
            or np.any(lower > 0) or np.any(upper < 0)):
        raise ValueError("expanded command bounds must contain zero in all three axes")
    # Largest-remainder allocation preserves the requested proportions in small batches.
    counts = np.floor(batch_size * weights).astype(int)
    counts[np.argsort(-(batch_size * weights - counts), kind="stable")[:batch_size-counts.sum()]] += 1
    labels = np.repeat(COMMAND_PROFILES, counts) if profile == "mixed" else np.full(batch_size, profile)
    rng.shuffle(labels)
    names = np.array(["forward", "backward", "left", "right", "turn_left", "turn_right", "combined", "standing"])
    directions = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                           [0, 0, 1], [0, 0, -1], [1, 1, .5], [0, 0, 0]])
    behavior = np.empty(batch_size, dtype=int)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        modes = np.arange(len(indices)) % len(names)
        rng.shuffle(modes)
        behavior[indices] = modes
    direction = directions[behavior]
    # Sample multiple speeds inside the configured box, never beyond policy limits.
    target = direction * rng.uniform(.25, 1., (batch_size, 1))
    target = np.where(target >= 0, target * upper, target * -lower)
    segment = np.arange(queries) // hold_steps
    levels = rng.uniform(.25, 1., (batch_size, segment[-1] + 2))
    levels[:, 1::3] = 0.  # Explicit stopping intervals, not just small random commands.
    command = np.empty((batch_size, queries, 3), dtype=np.float32)
    clock = np.arange(queries) * control_dt
    for index, label in enumerate(labels):
        if label == "constant_zoh":
            values = np.broadcast_to(target[index], (queries, 3))
        elif label in ("multi_step_zoh", "smooth"):
            gain = levels[index, segment]
            if label == "smooth":
                alpha = (np.arange(queries) % hold_steps) / hold_steps
                alpha = alpha**2 * (3. - 2. * alpha)
                previous = np.where(segment > 0, levels[index, np.maximum(segment-1, 0)], 0.)
                gain = previous + alpha * (gain - previous)
            values = gain[:, None] * target[index]
        elif label == "multisine":
            frequency = rng.uniform(.15, .6, (2, 3))
            phase = rng.uniform(-np.pi, np.pi, (2, 3))
            wave = np.sin(2*np.pi*clock[:, None, None]*frequency + phase)
            envelope = np.minimum(clock / .5, np.maximum(0., clock[-1] - clock) / .5).clip(0., 1.)
            values = target[index] * (wave[:, 0] * .65 + wave[:, 1] * .35) * envelope[:, None]
        else:
            gain = np.array([1., -1., 0.])[segment % 3]
            values = gain[:, None] * target[index]
        command[index] = np.clip(values, lower, upper)
    return command, labels, names[behavior]


############################################################################
# PARALLEL SIM CONFIG
############################################################################

@dataclass
class ParallelSimConfig:

    # simulation parameters
    batch_size: int                            # number of parallel environments
    seed: int                                  # random seed
    
    # model and policy paths
    xml_path: str                              # path to MuJoCo XML
    policy_path: str                           # path to .pt checkpoint
    sim_dt_des: float = None                   # desired sim timestep (None = use XML default)
    robot_config: G1Config = None              # robot config (Kp, Kd, etc.)
    
    # logging
    log_dtype: torch.dtype = torch.float32     # data type for logging
    log_hz: float = None                       # logging rate [Hz], must divide the sim rate (None = every sim step)

    # initial state bounds (if None, defaults to robot config default pose and zero velocity)
    q_lb: np.ndarray = None                    # lower bound on initial qpos, shape (nq,)
    q_ub: np.ndarray = None                    # upper bound on initial qpos, shape (nq,)
    v_lb: np.ndarray = None                    # lower bound on initial qvel, shape (nv,)
    v_ub: np.ndarray = None                    # upper bound on initial qvel, shape (nv,)

    # cmd randomization: resample per-env cmd uniformly from [cmd_lb, cmd_ub], 
    cmd_zoh_steps: int = None                  # control queries per ZOH window (defaults to new command at every policy query)
    cmd_lb: np.ndarray = None                  # cmd sample lower bound, shape (3,); clipped to [-cmd_scale, +cmd_scale]; default = -cmd_scale
    cmd_ub: np.ndarray = None                  # cmd sample upper bound, shape (3,); clipped to [-cmd_scale, +cmd_scale]; default = +cmd_scale
    command_profile: str = "zoh"               # original uniform ZOH, or expanded profiles
    profile_weights: tuple = PROFILE_WEIGHTS


############################################################################
# PARALLEL SIM CLASS
############################################################################

class ParallelSim:
    """Parallel G1 rollout using MuJoCo Warp + PyTorch policy."""

    def __init__(self, config: ParallelSimConfig):

        # use GPU for simulation and policy inference
        self.device = torch.device("cuda:0")

        # config parameters
        self.batch_size = config.batch_size
        self.log_dtype = config.log_dtype
        self.command_profile = config.command_profile
        self.profile_weights = config.profile_weights
        self.profile_rng = np.random.default_rng(config.seed)
        if self.batch_size < 1 or self.command_profile not in ("zoh", "mixed", *COMMAND_PROFILES):
            raise ValueError("positive batch_size and a recognized command_profile are required")

        # random seed
        seed = config.seed
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        print(f"Random seed: {seed}")

        # robot config
        self.robot_cfg = config.robot_config or G1Config()
        self.nu = len(self.robot_cfg.Kp)  # 23

        # PD gains and default positions on GPU
        self.Kp = torch.tensor(self.robot_cfg.Kp, dtype=torch.float32, device=self.device)
        self.Kd = torch.tensor(self.robot_cfg.Kd, dtype=torch.float32, device=self.device)
        self.default_joint_pos = torch.tensor(
            self.robot_cfg.default_joint_pos, dtype=torch.float32, device=self.device
        )
        self.action_scale = torch.tensor(
            self.robot_cfg.action_scale, dtype=torch.float32, device=self.device
        )
        self.cmd_scale = torch.tensor(
            self.robot_cfg.cmd_scale, dtype=torch.float32, device=self.device
        )  # (3,) clamp limit for cmd sample bounds (default box = ±cmd_scale)

        # per-dim cmd sample bounds (clipped to [-cmd_scale, +cmd_scale])
        if config.cmd_lb is None:
            cmd_lb_t = -self.cmd_scale.clone()
        else:
            cmd_lb_t = torch.tensor(
                np.asarray(config.cmd_lb, dtype=np.float32),
                dtype=torch.float32, device=self.device,
            )
            if cmd_lb_t.shape != (3,):
                raise ValueError(f"cmd_lb must have shape (3,), got {tuple(cmd_lb_t.shape)}")
        if config.cmd_ub is None:
            cmd_ub_t = +self.cmd_scale.clone()
        else:
            cmd_ub_t = torch.tensor(
                np.asarray(config.cmd_ub, dtype=np.float32),
                dtype=torch.float32, device=self.device,
            )
            if cmd_ub_t.shape != (3,):
                raise ValueError(f"cmd_ub must have shape (3,), got {tuple(cmd_ub_t.shape)}")
        self.cmd_lb = torch.clamp(cmd_lb_t, -self.cmd_scale, +self.cmd_scale)
        self.cmd_ub = torch.clamp(cmd_ub_t, -self.cmd_scale, +self.cmd_scale)
        if not bool(torch.isfinite(self.cmd_lb).all() & torch.isfinite(self.cmd_ub).all()
                    & (self.cmd_lb <= self.cmd_ub).all()):
            raise ValueError("command bounds must be finite and ordered")

        # writable (B, 3) cmd buffer — repeated rather than expanded so we can resample it
        self.cmd = torch.tensor(
            self.robot_cfg.cmd, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(self.batch_size, 1).contiguous()  # (B, 3)
        # None defaults to 1: resample cmd at every control query
        self.cmd_zoh_steps = 1 if config.cmd_zoh_steps is None else int(config.cmd_zoh_steps)
        if self.cmd_zoh_steps < 1:
            raise ValueError("cmd_zoh_steps must be positive")

        # timing
        self.control_dt = self.robot_cfg.control_dt
        self.gait_period = self.robot_cfg.gait_period

        # load mujoco model and data
        self.mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # override sim timestep if desired (this is the timestep used for physics integration)
        if config.sim_dt_des is not None:
            self.mj_model.opt.timestep = config.sim_dt_des
        self.sim_dt = float(self.mj_model.opt.timestep)

        # compute control decimation (number of sim steps per control step)
        control_ratio = self.control_dt / self.sim_dt
        if not math.isclose(control_ratio, round(control_ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"sim_dt ({self.sim_dt}) must divide control_dt ({self.control_dt}) exactly."
            )
        self.decimation = int(round(control_ratio))

        # compute log decimation (number of sim steps per logged sample)
        log_hz = 1.0 / self.sim_dt if config.log_hz is None else config.log_hz
        log_ratio = 1.0 / (log_hz * self.sim_dt) if log_hz > 0 else 0.0
        if log_ratio < 1 or not math.isclose(log_ratio, round(log_ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"log_hz ({log_hz}) must divide the sim rate ({1.0 / self.sim_dt:g} Hz) exactly."
            )
        self.log_decimation = int(round(log_ratio))
        self.log_dt = self.log_decimation * self.sim_dt

        # robot state size
        self.nq = self.mj_model.nq  # 30  (7 base + 23 joints)
        self.nv = self.mj_model.nv  # 29  (6 base + 23 joints)

        # initial condition state bounds
        default_q = np.zeros(self.nq, dtype=np.float32)
        default_q[:7] = self.robot_cfg.default_base_pos
        default_q[7:7 + self.nu] = self.robot_cfg.default_joint_pos
        default_v = np.zeros(self.nv, dtype=np.float32)
        q_lb = default_q if config.q_lb is None else np.asarray(config.q_lb, dtype=np.float32)
        q_ub = default_q if config.q_ub is None else np.asarray(config.q_ub, dtype=np.float32)
        v_lb = default_v if config.v_lb is None else np.asarray(config.v_lb, dtype=np.float32)
        v_ub = default_v if config.v_ub is None else np.asarray(config.v_ub, dtype=np.float32)

        # ensure that bounds have correct shape
        if q_lb.shape != (self.nq,) or q_ub.shape != (self.nq,):
            raise ValueError(f"q_lb and q_ub must each have shape ({self.nq},)")
        if v_lb.shape != (self.nv,) or v_ub.shape != (self.nv,):
            raise ValueError(f"v_lb and v_ub must each have shape ({self.nv},)")
        
        self.q_lb = torch.tensor(q_lb, dtype=torch.float32, device=self.device)
        self.q_ub = torch.tensor(q_ub, dtype=torch.float32, device=self.device)
        self.v_lb = torch.tensor(v_lb, dtype=torch.float32, device=self.device)
        self.v_ub = torch.tensor(v_ub, dtype=torch.float32, device=self.device)

        # set default initial state
        default_base = self.robot_cfg.default_base_pos
        default_joints = self.robot_cfg.default_joint_pos
        self.mj_data.qpos[:7] = default_base
        self.mj_data.qpos[7:7 + self.nu] = default_joints

        # put model and data on GPU
        with wp.ScopedDevice("cuda:0"):
            self.wp_model = mjwarp.put_model(self.mj_model)
            self.wp_data = mjwarp.put_data(
                self.mj_model, self.mj_data, nworld=self.batch_size,
                njmax=200, nconmax=200,
            )

        # zero-copy torch views into warp arrays
        self.t_qpos = wp.to_torch(self.wp_data.qpos)        # (B, nq)
        self.t_qvel = wp.to_torch(self.wp_data.qvel)        # (B, nv)
        self.t_ctrl = wp.to_torch(self.wp_data.ctrl)        # (B, nu)
        self.t_sensordata = wp.to_torch(self.wp_data.sensordata)  # (B, nsensordata)
        self.t_time = wp.to_torch(self.wp_data.time)        # (B,)

        print(f"MuJoCo Warp initialized: nq={self.nq}, nv={self.nv}, nu={self.nu}")
        print(f"  sim_dt=[{self.sim_dt} sec]")
        print(f"  control_dt=[{self.control_dt} sec]")
        print(f"  control decimation=[{self.decimation}]")
        print(f"  log_dt=[{self.log_dt} sec] (log decimation=[{self.log_decimation}])")
        print(f"  batch_size=[{self.batch_size}]")

        # load policy
        self.policy = load_policy(config.policy_path, self.device)
        print(f"Loaded policy from [{config.policy_path}].")

    # ------------------------------------------------------------------ #
    #  Observation
    # ------------------------------------------------------------------ #

    def build_observation(self, prev_action: torch.Tensor, sim_time: torch.Tensor) -> torch.Tensor:
        """
        Build the 80-dim observation vector from simulation state.

        Args:
            prev_action: (B, 23) previous policy output
            sim_time:    (B,)    simulation time per env
        Returns:
            obs: (B, 80)
        """
        # from sensordata: pelvis gyro (body-frame angular velocity)
        # sensor layout: torso_quat(4), torso_gyro(3), torso_acc(3),   [0:10]
        #                pelvis_quat(4), pelvis_gyro(3), pelvis_acc(3) [10:20]
        #                left_foot_touch(1), right_foot_touch(1)       [20:22]
        omega_body = self.t_sensordata[:, 14:17]       # (B, 3)

        # base quaternion for gravity projection
        quat = self.t_qpos[:, 3:7]                     # (B, 4)
        gravity = get_gravity_orientation_batch(quat)   # (B, 3)

        # joint state
        qpos_joints = self.t_qpos[:, 7:7 + self.nu]   # (B, 23)
        qvel_joints = self.t_qvel[:, 6:6 + self.nu]   # (B, 23)

        gait_phase = self.gait_phase(sim_time)

        # joint position error
        qj_err = qpos_joints - self.default_joint_pos  # (B, 23)

        # [omega(3), gravity(3), cmd(3), phase(2), qj_err(23), dqj(23), prev_action(23)]
        return torch.cat([omega_body, gravity, self.cmd, gait_phase,
                          qj_err, qvel_joints, prev_action], dim=-1)

    def gait_phase(self, sim_time):
        """The unchanged policy clock, also recorded alongside the full state."""
        phase = (sim_time % self.gait_period) / self.gait_period  # (B,)
        two_pi = 2.0 * math.pi
        gait_sin = torch.sin(two_pi * phase)   # (B,)
        gait_cos = torch.cos(two_pi * phase)   # (B,)
        gait_phase = torch.stack([gait_sin, gait_cos], dim=-1)  # (B, 2)

        # zero gait phase when command is below threshold
        cmd_norm = torch.norm(self.cmd, dim=-1)  # (B,)
        standing = (cmd_norm < self.robot_cfg.stand_cmd_threshold)
        gait_phase[standing] = 0.0

        return gait_phase

    # ------------------------------------------------------------------ #
    #  PD Control
    # ------------------------------------------------------------------ #

    def compute_torque(self, action: torch.Tensor) -> torch.Tensor:
        """
        PD control: action → desired position → torque.

        Args:
            action: (B, 23) raw policy output
        Returns:
            tau: (B, 23)
        """
        qpos_des = action * self.action_scale + self.default_joint_pos
        qpos_joints = self.t_qpos[:, 7:7 + self.nu]
        qvel_joints = self.t_qvel[:, 6:6 + self.nu]
        tau = self.Kp * (qpos_des - qpos_joints) + self.Kd * (0.0 - qvel_joints)
        return tau

    # ------------------------------------------------------------------ #
    #  Initial Conditions
    # ------------------------------------------------------------------ #

    def sample_random_uniform_initial_conditions(self):
        """
        Sample initial conditions uniformly from configured q/v bounds.

        Returns:
            q0_batch: (B, nq) initial positions
            v0_batch: (B, nv) initial velocities
        """
        B = self.batch_size

        q_rand = torch.rand(B, self.nq, dtype=torch.float32, device=self.device, generator=self.rng)
        v_rand = torch.rand(B, self.nv, dtype=torch.float32, device=self.device, generator=self.rng)

        q0_batch = self.q_lb.unsqueeze(0) + (self.q_ub - self.q_lb).unsqueeze(0) * q_rand
        v0_batch = self.v_lb.unsqueeze(0) + (self.v_ub - self.v_lb).unsqueeze(0) * v_rand

        return q0_batch, v0_batch

    # ------------------------------------------------------------------ #
    #  Command Sampling
    # ------------------------------------------------------------------ #

    def _sample_cmd(self):
        """Resample per-env cmd uniformly from box [cmd_lb, cmd_ub]."""
        u = torch.rand(self.batch_size, 3, dtype=torch.float32,
                       device=self.device, generator=self.rng)
        self.cmd = self.cmd_lb.unsqueeze(0) + (self.cmd_ub - self.cmd_lb).unsqueeze(0) * u  # (B, 3)

    # ------------------------------------------------------------------ #
    #  Rollout
    # ------------------------------------------------------------------ #

    def rollout_policy_input(self, T: int):
        """
        Sample random ICs and run a policy rollout.

        Args:
            T: number of logged samples, incl. the initial state, log_dt apart
        Returns:
            q_log:   (B, T, nq) joint positions
            v_log:   (B, T, nv) joint velocities
            cmd_log: (B, T, 3)  per-env cmd in effect at each logged timestep
        """
        q0_batch, v0_batch = self.sample_random_uniform_initial_conditions()
        return self._rollout_policy_input(q0_batch, v0_batch, T)

    def _rollout_policy_input(self, q0_batch: torch.Tensor, v0_batch: torch.Tensor, T: int):
        """Run the policy from given ICs for (T-1)*log_decimation sim steps, logging T samples.

        Args:
            q0_batch: (B, nq) initial positions
            v0_batch: (B, nv) initial velocities
            T:        number of logged samples, incl. the initial state
        Returns:
            q_log, v_log: (B, T, nq), (B, T, nv) joint positions, velocities (logged at log_dt)
            cmd_log:      (B, T, 3) cmd active at each logged timestep
        """
        B = self.batch_size
        if not isinstance(T, int) or T < 2:
            raise ValueError("T must include at least two logged samples")
        S = (T - 1) * self.log_decimation  # number of sim steps to take (T includes initial state)
        queries = (S + self.decimation - 1) // self.decimation
        schedule = None
        if self.command_profile != "zoh":
            schedule, labels, behaviors = make_command_schedule(
                B, queries, self.cmd_zoh_steps, self.control_dt,
                self.cmd_lb.cpu().numpy(), self.cmd_ub.cpu().numpy(), self.profile_rng,
                self.command_profile, self.profile_weights,
            )
            schedule = torch.as_tensor(schedule, device=self.device)
        else:
            labels, behaviors = np.full(B, "zoh"), np.full(B, "uniform_box")

        # reset data to default state
        mjwarp.reset_data(self.wp_model, self.wp_data)

        # set initial conditions from inputs
        self.t_qpos[:] = q0_batch
        self.t_qvel[:] = v0_batch

        # run forward to initialize sensors/kinematics
        with wp.ScopedDevice("cuda:0"):
            mjwarp.forward(self.wp_model, self.wp_data)

        # logging at log_dt
        q_log = torch.zeros(B, T, self.nq, dtype=self.log_dtype, device=self.device)
        v_log = torch.zeros(B, T, self.nv, dtype=self.log_dtype, device=self.device)
        cmd_log = torch.zeros(B, T, 3, dtype=self.log_dtype, device=self.device)
        # Additive diagnostics; preserve the public (q_log, v_log, cmd_log) return.
        self.rollout_logs = {
            name: torch.empty(B, count, width, dtype=self.log_dtype, device=self.device)
            for name, count, width in (
                ("prev_action_log", T, self.nu), ("gait_phase_log", T, 2),
                ("foot_force_log", T, 2), ("policy_cmd_log", queries, 3),
                ("policy_prev_action_log", queries, self.nu),
                ("policy_action_log", queries, self.nu), ("policy_gait_phase_log", queries, 2),
            )
        }
        self.rollout_logs["time_log"] = torch.empty(B, T, dtype=torch.float64, device=self.device)
        self.rollout_logs["policy_time_log"] = torch.empty(B, queries, dtype=torch.float64, device=self.device)
        self.profile_labels, self.behavior_labels = labels, behaviors

        # sample initial cmd
        if schedule is None:
            self._sample_cmd()
        else:
            self.cmd[:] = schedule[:, 0]

        # log initial state
        q_log[:, 0, :] = self.t_qpos.to(self.log_dtype)
        v_log[:, 0, :] = self.t_qvel.to(self.log_dtype)
        cmd_log[:, 0, :] = self.cmd.to(self.log_dtype)

        # initialize previous action
        prev_action = torch.zeros(B, self.nu, dtype=torch.float32, device=self.device)
        action = prev_action
        self._log_memory(0, prev_action)

        # main rollout loop
        ctrl_query = 0
        with wp.ScopedDevice("cuda:0"):
            for step in range(S):
                # update policy every decimation steps
                if step % self.decimation == 0:
                    # resample cmd on ZOH boundaries (skip query 0 — already sampled above)
                    if schedule is not None:
                        self.cmd[:] = schedule[:, ctrl_query]
                    elif ctrl_query > 0 and ctrl_query % self.cmd_zoh_steps == 0:
                        self._sample_cmd()

                    sim_time = self.t_time  # (B,)
                    obs = self.build_observation(prev_action, sim_time)
                    self.rollout_logs["policy_time_log"][:, ctrl_query] = sim_time
                    self.rollout_logs["policy_cmd_log"][:, ctrl_query] = self.cmd
                    self.rollout_logs["policy_prev_action_log"][:, ctrl_query] = prev_action
                    self.rollout_logs["policy_gait_phase_log"][:, ctrl_query] = obs[:, 9:11]
                    action = self.policy(obs)  # (B, 23)
                    self.rollout_logs["policy_action_log"][:, ctrl_query] = action
                    prev_action = action
                    ctrl_query += 1

                # PD torque and physics step
                tau = self.compute_torque(action)
                self.t_ctrl[:] = tau
                mjwarp.step(self.wp_model, self.wp_data)
                # step() evaluates sensors BEFORE integration. Associate touch
                # forces with the pre-step snapshot, not the new qpos/time.
                if step % self.log_decimation == 0:
                    self.rollout_logs["foot_force_log"][:, step // self.log_decimation] = self.t_sensordata[:, 20:22]

                # log state every log_decimation sim steps
                if (step + 1) % self.log_decimation == 0:
                    k = (step + 1) // self.log_decimation
                    q_log[:, k, :] = self.t_qpos.to(self.log_dtype)
                    v_log[:, k, :] = self.t_qvel.to(self.log_dtype)
                    cmd_log[:, k, :] = self.cmd.to(self.log_dtype)
                    self._log_memory(k, prev_action)

            # Only the terminal diagnostic needs another forward pass. No
            # subsequent physics/policy step sees its solver or sensor state.
            self.t_ctrl[:] = self.compute_torque(action)
            mjwarp.forward(self.wp_model, self.wp_data)
            self.rollout_logs["foot_force_log"][:, -1] = self.t_sensordata[:, 20:22]

        return q_log, v_log, cmd_log

    def _log_memory(self, index, previous_action):
        self.rollout_logs["time_log"][:, index] = self.t_time
        self.rollout_logs["prev_action_log"][:, index] = previous_action
        self.rollout_logs["gait_phase_log"][:, index] = self.gait_phase(self.t_time)

    def rollout_policy_input_x0(self, x0_batch: torch.Tensor, T: int):
        """Run a policy rollout from a stacked state vector x0 = [q, v].

        Args:
            x0_batch: (B, nq+nv) initial states
            T:        number of logged samples, incl. the initial state, log_dt apart
        Returns:
            q_log:   (B, T, nq) joint positions
            v_log:   (B, T, nv) joint velocities
            cmd_log: (B, T, 3)  cmd active at each timestep
        """
        q0_batch = x0_batch[:, :self.nq]
        v0_batch = x0_batch[:, self.nq:self.nq + self.nv]
        return self._rollout_policy_input(q0_batch, v0_batch, T)


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    # parse command-line args
    parser = argparse.ArgumentParser(description="Generate G1 rollout datasets on the GPU.")
    parser.add_argument("--log_hz", type=float, default=None, metavar="HZ",
                        help="logging rate; must divide the sim rate (default: every sim step)")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=1., help="trajectory length in seconds")
    parser.add_argument("--datasets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--cmd-zoh-steps", type=int, default=50, help="policy queries per command segment")
    parser.add_argument("--command-profile", choices=("zoh", "mixed", *COMMAND_PROFILES), default="zoh")
    parser.add_argument("--profile-weights", type=float, nargs=5, default=PROFILE_WEIGHTS,
                        metavar="W", help="constant, multi-step, smooth, multisine, reversal proportions")
    parser.add_argument("--command-scale", type=float, default=1., help="fraction of existing policy command limits (0, 1]")
    parser.add_argument("--output-dir", type=Path, help="override the default data/data destination")
    args = parser.parse_args()
    if args.batch_size < 1 or args.datasets < 1 or not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("batch size, datasets, and duration must be positive")
    if not 0 < args.command_scale <= 1 or args.cmd_zoh_steps < 1:
        parser.error("command scale must be in (0, 1] and ZOH steps must be positive")
    legacy_dir = Path(ROOT_DIR) / "data" / "data"

    # print the device being used
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # paths
    xml_path = os.path.join(ROOT_DIR, "models", "g1_23dof.xml")
    policy_path = os.path.join(ROOT_DIR, "policy", "g1_23dof_vel.pt")

    # rollout parameters
    batch_size = args.batch_size
    sim_dt_des = 0.002  # dt for simulation (too coarse causes sim instability) - DO NOT CHANGE!!
    duration = args.duration  # old 1-second default retained; use longer rollouts for training windows

    # number of datasets to generate
    N_datasets = args.datasets

    # cmd randomization: resample per-env cmd every N policy queries (None = resample every control query)
    # at control_dt=0.02 s, 125 queries = 2.5 s of hold per ZOH window
    cmd_zoh_steps = args.cmd_zoh_steps

    # random /fixed seed
    seed = args.seed
    #seed = int(time.time())

    # create the config for parallel sim
    config = ParallelSimConfig(
        batch_size=batch_size,
        xml_path=xml_path,
        policy_path=policy_path,
        seed=seed,
        sim_dt_des=sim_dt_des,
        cmd_zoh_steps=cmd_zoh_steps,
        log_hz=args.log_hz,
        cmd_lb=-G1Config().cmd_scale * args.command_scale,
        cmd_ub=G1Config().cmd_scale * args.command_scale,
        command_profile=args.command_profile,
        profile_weights=tuple(args.profile_weights),
    )

    # create the rollout instance; T counts logged samples, incl. the initial state
    r = ParallelSim(config)
    T = int(round(duration / r.log_dt)) + 1
    if T < 2 or not math.isclose((T-1)*r.log_dt, duration, abs_tol=1e-9):
        parser.error("duration must be a positive multiple of the logging period")
    metadata = dict(
        schema_version=2, command_profile=args.command_profile,
        profile_weights=list(args.profile_weights),
        cmd_lb=r.cmd_lb.cpu().tolist(), cmd_ub=r.cmd_ub.cpu().tolist(),
        robot_config={key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in vars(r.robot_cfg).items()},
        initial_bounds={name: getattr(r, name).cpu().tolist() for name in ("q_lb", "q_ub", "v_lb", "v_ub")},
        sha256={name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in (("model", xml_path), ("policy", policy_path), ("generator", __file__))},
        versions=dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                      mujoco=mujoco.__version__, mujoco_warp=mjwarp.__version__, warp=wp.__version__),
        device=torch.cuda.get_device_name(0), coordinates="world qpos; world linear/body angular qvel; body commands",
        alignment="state/time/previous-action sampled before the next query; cmd_log is the incoming held command; policy_cmd_log is outgoing at policy_time_log",
        contact="left/right foot touch forces evaluated at the logged state (pre-integration); contact flag is force > 0",
    )

    # function to rollout and save full-order + ROM datasets,
    # returns time it took to complete the rollout (excluding saving time)
    def rollout_and_save_dataset(save_path, rom_save_path, label):

        print(f"\nGenerating {label}...")

        t0 = time.time()
        rng_state = r.rng.get_state().cpu().numpy()
        profile_rng_state = json.dumps(r.profile_rng.bit_generator.state)
        q_log, v_log, cmd_log = r.rollout_policy_input(T)
        torch.cuda.synchronize()
        rollout_time = time.time() - t0
        print(f"Rollout time: {rollout_time:.3f}s")

        q_log_np = q_log.cpu().numpy()
        v_log_np = v_log.cpu().numpy()
        cmd_log_np = cmd_log.cpu().numpy()
        extra_logs = {name: values.cpu().numpy() for name, values in r.rollout_logs.items()}
        extra_logs["foot_contact_log"] = extra_logs["foot_force_log"] > 0.
        provenance = dict(
            command_profile_labels=r.profile_labels, locomotion_labels=r.behavior_labels,
            generation_metadata=json.dumps(metadata), dataset_index=i+1,
            torch_rng_state_before=rng_state, profile_rng_state_before=profile_rng_state,
        )

        # derive ROM state: q_rom = [px, py, theta] (world frame),
        # v_rom = [vx, vy, omega_z] (body frame, matches body-frame cmd input)
        q_rom_log_np = q_to_rom(q_log_np)              # (B, T, 3)
        v_rom_log_np = v_to_rom(q_log_np, v_log_np)    # (B, T, 3)

        print(f"  q_log:       {q_log_np.shape}")
        print(f"  v_log:       {v_log_np.shape}")
        print(f"  cmd_log:     {cmd_log_np.shape}")
        print(f"  q_rom_log:   {q_rom_log_np.shape}")
        print(f"  v_rom_log:   {v_rom_log_np.shape}")

        # full-order save
        np.savez(
            save_path,
            sim_dt=float(r.sim_dt),
            control_dt=float(r.control_dt),
            log_dt=float(r.log_dt),
            batch_size=int(batch_size),
            T=int(T),
            N_datasets=int(N_datasets),
            cmd_zoh_steps=int(cmd_zoh_steps),
            seed=int(seed),
            q_log=q_log_np,
            v_log=v_log_np,
            cmd_log=cmd_log_np,
            **extra_logs, **provenance,
        )
        print(f"  Saved full-order to: {save_path}")

        # ROM save
        np.savez(
            rom_save_path,
            sim_dt=float(r.sim_dt),
            control_dt=float(r.control_dt),
            log_dt=float(r.log_dt),
            batch_size=int(batch_size),
            T=int(T),
            N_datasets=int(N_datasets),
            cmd_zoh_steps=int(cmd_zoh_steps),
            seed=int(seed),
            q_rom_log=q_rom_log_np,
            v_rom_log=v_rom_log_np,
            cmd_log=cmd_log_np,
            **provenance,
        )
        print(f"  Saved ROM to:        {rom_save_path}")

        return rollout_time

    # Profile-specific files stay separate, but match the training config's
    # g1_23dof_data_*.npz glob in data/data by default.

    tot_time = 0.0
    save_dir = str(args.output_dir or legacy_dir)
    os.makedirs(save_dir, exist_ok=True)
    # Never overwrite an existing full-order OR ROM recording, including ZOH.
    for number in range(1, N_datasets+1):
        for target in dataset_paths(save_dir, args.command_profile, batch_size, T, cmd_zoh_steps, seed, number):
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"refusing to overwrite existing recording: {target}")

    for i in range(N_datasets):
        save_path, rom_save_path = dataset_paths(
            save_dir, args.command_profile, batch_size, T, cmd_zoh_steps, seed, i+1,
        )

        tot_time += rollout_and_save_dataset(
            save_path,
            rom_save_path,
            f"dataset {i+1}/{N_datasets}",
        )
    print(f"\nFinished generating {N_datasets} datasets.")
    print(f"Total rollout time: {tot_time:.1f}s")
    print(f"Total trajectories: {N_datasets * batch_size}")
