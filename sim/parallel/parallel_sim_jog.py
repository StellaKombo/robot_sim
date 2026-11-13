##
#
#  Parallel G1 29-DoF JOG simulation using MuJoCo Warp (GPU) + a torch jog policy.
#  Same rollout / logging / dataset layout as parallel_sim_walk.py (23-DoF walk), but with the
#  jog policy's 98-D observation, 43-frame gait clock and a mode-based command sampler.
#
##

# standard imports
import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field

# torch, mujoco, warp imports
import numpy as np
import torch
import mujoco
import mujoco_warp as mjwarp
import warp as wp

# custom imports
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)
from policy.jog.jog_actor import JogActor
from utils.math_utils import get_gravity_orientation_batch, q_to_rom, v_to_rom


# command modes; the policy's trained twists are disjoint per mode, not a box
MODE_STAND, MODE_FWD, MODE_BWD, MODE_PIVOT = 0, 1, 2, 3
MODE_NAMES = ["stand", "forward", "backward", "pivot"]


############################################################################
# PARALLEL SIM CONFIG
############################################################################

@dataclass
class JogSimConfig:

    # simulation parameters
    batch_size: int                            # number of parallel environments
    seed: int                                  # random seed
    xml_path: str                              # path to the 29-DoF MuJoCo model (robot + scene)
    policy_path: str                           # path to the jog ONNX export (run with onnxruntime)
    sim_dt_des: float = 0.002                  # physics timestep [s]
    log_dtype: torch.dtype = torch.float32     # data type for logging
    log_hz: float = None                       # logging rate [Hz], must divide the sim rate (None = every sim step)

    # rollout structure
    stand_s: float = 1.0                       # [s] zero command at the start of every rollout
    cmd_zoh_steps: int = 125                   # control queries per command hold (125 = 2.5 s)
    random_phase: bool = True                  # random gait-clock offset per env (training starts mid-clip)
    init_joint_noise: float = 0.0              # [rad] uniform +- noise on the initial joint pose
    fall_height: float = 0.45                  # [m] pelvis height below which an env is flagged fallen

    # mode mix, order: stand, forward, backward, pivot
    mode_weights: tuple = (0.20, 0.35, 0.25, 0.20)

    # trained twist bands (deploy_robot g1_29dof_jog_unicycle_lstm.yaml)
    fwd_vx: tuple = (0.50, 1.50)
    bwd_vx: tuple = (-1.00, -0.50)
    curve_wz: tuple = (-0.50, 0.50)            # yaw rate while driving forward / backward
    pivot_wz: tuple = (1.00, 2.00)             # in-place yaw-rate magnitude, sign random


############################################################################
# PARALLEL SIM CLASS
############################################################################

class ParallelJogSim:
    """Parallel G1 29-DoF jog rollouts using MuJoCo Warp + a batched torch jog policy."""

    def __init__(self, config: JogSimConfig):

        self.cfg = config
        self.device = torch.device("cuda:0")
        self.batch_size = config.batch_size
        self.log_dtype = config.log_dtype
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(config.seed)
        print(f"Random seed: {config.seed}")

        # policy and its deployment params (from the ONNX metadata)
        self.policy = JogActor(config.policy_path)
        self.policy_name = os.path.basename(config.policy_path)
        meta = self.policy.meta
        f32 = dict(dtype=torch.float32, device=self.device)
        self.default_joint_pos = torch.tensor(meta["default_joint_pos"], **f32)
        self.action_scale = torch.tensor(meta["action_scale"], **f32)
        self.Kp = torch.tensor(meta["joint_stiffness"], **f32)
        self.Kd = torch.tensor(meta["joint_damping"], **f32)
        self.period = int(meta["motion_period_frames"])
        self.nu = len(meta["default_joint_pos"])
        expected = ["base_ang_vel", "joint_pos", "joint_vel", "actions",
                    "commanded_twist", "motion_phase", "projected_gravity"]
        assert meta["observation_names"] == expected, f"unexpected obs layout {meta['observation_names']}"
        print(f"Loaded {self.policy.kind} jog policy [{self.policy_name}] "
              f"from [{config.policy_path}], gait period {self.period} frames.")

        # mode sampler tensors
        w = torch.tensor(config.mode_weights, **f32)
        self.mode_probs = w / w.sum()
        self.cmd = torch.zeros(self.batch_size, 3, **f32)
        self.mode = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # mujoco model
        self.mj_model = mujoco.MjModel.from_xml_path(config.xml_path)
        self.mj_model.opt.timestep = config.sim_dt_des
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sim_dt = float(self.mj_model.opt.timestep)
        self.nq, self.nv = self.mj_model.nq, self.mj_model.nv
        assert self.mj_model.nu == self.nu, f"model has {self.mj_model.nu} actuators, policy {self.nu}"

        # joint order: actuators must be the policy's joints, in order
        act_joints = [self.mj_model.joint(self.mj_model.actuator_trnid[i, 0]).name for i in range(self.nu)]
        assert act_joints == list(meta["joint_names"]), "actuator order differs from the policy's joint order"

        # timing: control and log decimation
        self.control_dt = 0.02
        self.decimation = self._ratio(self.control_dt / self.sim_dt, "control_dt", "sim_dt")
        log_hz = 1.0 / self.sim_dt if config.log_hz is None else config.log_hz
        self.log_decimation = self._ratio(1.0 / (log_hz * self.sim_dt), f"log_hz {log_hz}", "sim rate")
        self.log_dt = self.log_decimation * self.sim_dt
        self.stand_queries = int(round(config.stand_s / self.control_dt))

        # pelvis IMU slots in sensordata (same sensors deploy_robot reads)
        self.gyro_adr = int(self.mj_model.sensor("pelvis_imu_gyro_sensor").adr[0])
        self.quat_adr = int(self.mj_model.sensor("pelvis_imu_quat_sensor").adr[0])

        # initial state: the policy's standing pose
        self.default_q = np.zeros(self.nq, dtype=np.float32)
        self.default_q[:7] = [0.0, 0.0, 0.776, 1.0, 0.0, 0.0, 0.0]
        self.default_q[7:7 + self.nu] = meta["default_joint_pos"]
        self.mj_data.qpos[:] = self.default_q

        # put model and data on GPU
        with wp.ScopedDevice("cuda:0"):
            self.wp_model = mjwarp.put_model(self.mj_model)
            self.wp_data = mjwarp.put_data(self.mj_model, self.mj_data, nworld=self.batch_size,
                                           njmax=300, nconmax=200)

        # zero-copy torch views into warp arrays
        self.t_qpos = wp.to_torch(self.wp_data.qpos)              # (B, nq)
        self.t_qvel = wp.to_torch(self.wp_data.qvel)              # (B, nv)
        self.t_ctrl = wp.to_torch(self.wp_data.ctrl)              # (B, nu)
        self.t_sensordata = wp.to_torch(self.wp_data.sensordata)  # (B, nsensordata)

        print(f"MuJoCo Warp initialized: nq={self.nq}, nv={self.nv}, nu={self.nu}")
        print(f"  sim_dt=[{self.sim_dt} sec], control_dt=[{self.control_dt} sec], "
              f"decimation=[{self.decimation}]")
        print(f"  log_dt=[{self.log_dt} sec] (log decimation=[{self.log_decimation}])")
        print(f"  batch_size=[{self.batch_size}]")
        print(f"  modes {dict(zip(MODE_NAMES, self.mode_probs.tolist()))}, "
              f"hold {config.cmd_zoh_steps * self.control_dt:.2f} s, stand first {config.stand_s} s")

    @staticmethod
    def _ratio(r, num, den):
        if r < 1 or not math.isclose(r, round(r), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"{den} must divide {num} exactly (ratio {r}).")
        return int(round(r))

    # ------------------------------------------------------------------ #
    #  Commands
    # ------------------------------------------------------------------ #

    def _uniform(self, lo, hi, n):
        u = torch.rand(n, device=self.device, generator=self.rng)
        return lo + (hi - lo) * u

    def _sample_cmd(self):
        """Pick a mode per env, then a twist inside that mode's trained band (vy = 0)."""
        B, c = self.batch_size, self.cfg
        self.mode = torch.multinomial(self.mode_probs, B, replacement=True, generator=self.rng)
        vx = torch.zeros(B, device=self.device)
        wz = torch.zeros(B, device=self.device)
        fwd, bwd, piv = self.mode == MODE_FWD, self.mode == MODE_BWD, self.mode == MODE_PIVOT
        vx = torch.where(fwd, self._uniform(*c.fwd_vx, B), vx)
        vx = torch.where(bwd, self._uniform(*c.bwd_vx, B), vx)
        wz = torch.where(fwd | bwd, self._uniform(*c.curve_wz, B), wz)
        sign = torch.where(torch.rand(B, device=self.device, generator=self.rng) < 0.5, -1.0, 1.0)
        wz = torch.where(piv, sign * self._uniform(*c.pivot_wz, B), wz)
        self.cmd = torch.stack([vx, torch.zeros_like(vx), wz], dim=-1)

    def _stand_cmd(self):
        self.cmd = torch.zeros(self.batch_size, 3, device=self.device)
        self.mode = torch.full((self.batch_size,), MODE_STAND, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------ #
    #  Observation and PD
    # ------------------------------------------------------------------ #

    def build_observation(self, prev_action: torch.Tensor, phase_step: torch.Tensor) -> torch.Tensor:
        """98-D jog observation (B, 98), in the policy's observation_names order."""
        omega = self.t_sensordata[:, self.gyro_adr:self.gyro_adr + 3]
        quat = self.t_sensordata[:, self.quat_adr:self.quat_adr + 4]
        qj = self.t_qpos[:, 7:7 + self.nu] - self.default_joint_pos
        dqj = self.t_qvel[:, 6:6 + self.nu]

        # gait clock (sin, cos) of 2*pi*t/T, blanked to (0, 0) on a zero command (idle stop)
        ang = 2.0 * math.pi * phase_step.float() / self.period
        phase = torch.stack([torch.sin(ang), torch.cos(ang)], dim=-1)
        moving = torch.any(self.cmd != 0.0, dim=-1, keepdim=True)
        phase = phase * moving

        return torch.cat([omega, qj, dqj, prev_action, self.cmd, phase,
                          get_gravity_orientation_batch(quat)], dim=-1)

    def compute_torque(self, action: torch.Tensor) -> torch.Tensor:
        qpos_des = action * self.action_scale + self.default_joint_pos
        qj = self.t_qpos[:, 7:7 + self.nu]
        dqj = self.t_qvel[:, 6:6 + self.nu]
        return self.Kp * (qpos_des - qj) - self.Kd * dqj

    # ------------------------------------------------------------------ #
    #  Rollout
    # ------------------------------------------------------------------ #

    def sample_initial_conditions(self):
        """Standing pose, plus optional uniform joint noise; zero velocity."""
        B = self.batch_size
        q0 = torch.tensor(self.default_q, device=self.device).unsqueeze(0).repeat(B, 1)
        if self.cfg.init_joint_noise > 0:
            n = self.cfg.init_joint_noise
            q0[:, 7:7 + self.nu] += self._uniform(-n, n, B * self.nu).view(B, self.nu)
        v0 = torch.zeros(B, self.nv, device=self.device)
        return q0, v0

    def rollout(self, T: int, fixed_cmd=None):
        """Run one batch for T logged samples (incl. the initial state).

        fixed_cmd: optional (3,) twist held for every env after the stand (for checks).
        Returns q_log (B,T,nq), v_log (B,T,nv), cmd_log (B,T,3), mode_log (B,T) int8,
        fell (B,) bool and fall_time (B,) [s] (NaN if it never fell).
        """
        B, S = self.batch_size, (T - 1) * self.log_decimation
        q0, v0 = self.sample_initial_conditions()

        mjwarp.reset_data(self.wp_model, self.wp_data)
        self.t_qpos[:] = q0
        self.t_qvel[:] = v0
        with wp.ScopedDevice("cuda:0"):
            mjwarp.forward(self.wp_model, self.wp_data)

        q_log = torch.zeros(B, T, self.nq, dtype=self.log_dtype, device=self.device)
        v_log = torch.zeros(B, T, self.nv, dtype=self.log_dtype, device=self.device)
        cmd_log = torch.zeros(B, T, 3, dtype=self.log_dtype, device=self.device)
        mode_log = torch.zeros(B, T, dtype=torch.int8, device=self.device)
        fell = torch.zeros(B, dtype=torch.bool, device=self.device)
        fall_time = torch.full((B,), float("nan"), device=self.device)

        self._stand_cmd()
        self.policy.reset(B)
        if self.cfg.random_phase:
            phase_step = torch.randint(0, self.period, (B,), device=self.device, generator=self.rng)
        else:
            phase_step = torch.zeros(B, dtype=torch.long, device=self.device)

        def log(k):
            q_log[:, k] = self.t_qpos.to(self.log_dtype)
            v_log[:, k] = self.t_qvel.to(self.log_dtype)
            cmd_log[:, k] = self.cmd.to(self.log_dtype)
            mode_log[:, k] = self.mode.to(torch.int8)

        log(0)
        action = torch.zeros(B, self.nu, device=self.device)
        ctrl_query = 0
        with wp.ScopedDevice("cuda:0"):
            for step in range(S):
                if step % self.decimation == 0:
                    # first command after the stand, then a new one every cmd_zoh_steps
                    q = ctrl_query - self.stand_queries
                    if q >= 0 and q % self.cfg.cmd_zoh_steps == 0:
                        if fixed_cmd is None:
                            self._sample_cmd()
                        else:
                            self.cmd = torch.tensor(fixed_cmd, dtype=torch.float32,
                                                    device=self.device).expand(B, 3).clone()
                            self.mode.fill_(MODE_FWD if fixed_cmd[0] > 0 else
                                            MODE_BWD if fixed_cmd[0] < 0 else
                                            MODE_PIVOT if fixed_cmd[2] != 0 else MODE_STAND)
                    obs = self.build_observation(action, phase_step)
                    action = self.policy(obs)
                    phase_step = (phase_step + 1) % self.period   # free-running, as in deploy
                    ctrl_query += 1

                self.t_ctrl[:] = self.compute_torque(action)
                mjwarp.step(self.wp_model, self.wp_data)

                if (step + 1) % self.log_decimation == 0:
                    log((step + 1) // self.log_decimation)
                    down = (self.t_qpos[:, 2] < self.cfg.fall_height) & ~fell
                    fall_time[down] = (step + 1) * self.sim_dt
                    fell |= down

        return q_log, v_log, cmd_log, mode_log, fell, fall_time


############################################################################
# MAIN
############################################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate G1 29-DoF jog rollout datasets on the GPU.")
    parser.add_argument("--log_hz", type=float, default=None, metavar="HZ",
                        help="logging rate; must divide the sim rate (default: every sim step)")
    args = parser.parse_args()
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # paths
    model_file = "g1_29dof.xml"
    xml_path = os.path.join(ROOT_DIR, "models", "g1_29dof", model_file)
    policy_path = os.path.join(ROOT_DIR, "policy", "jog", "g1_29dof_jog_lstm_v0_cont.onnx")

    # dataset names (distinct from the 23-DoF walk datasets)
    env_name = "g1_29dof_jog"
    rom_env_name = "g1_rom_jog"

    # rollout parameters
    batch_size = 1024
    duration = 10.0     # [s] rollout length: 1 s stand + 3-4 command holds
    N_datasets = 3
    seed = int(time.time())

    config = JogSimConfig(
        batch_size=batch_size,
        seed=seed,
        xml_path=xml_path,
        policy_path=policy_path,
        log_hz=args.log_hz,
    )
    r = ParallelJogSim(config)
    T = int(round(duration / r.log_dt)) + 1

    save_dir = os.path.join(ROOT_DIR, "data", "data")
    os.makedirs(save_dir, exist_ok=True)
    tot_time = 0.0
    for i in range(N_datasets):
        print(f"\nGenerating dataset {i+1}/{N_datasets}...")
        t0 = time.time()
        q_log, v_log, cmd_log, mode_log, fell, fall_time = r.rollout(T)
        torch.cuda.synchronize()
        rollout_time = time.time() - t0
        tot_time += rollout_time
        print(f"Rollout time: {rollout_time:.3f}s, fell: {int(fell.sum())}/{batch_size}")

        q_np, v_np = q_log.cpu().numpy(), v_log.cpu().numpy()
        common = dict(sim_dt=float(r.sim_dt), control_dt=float(r.control_dt), log_dt=float(r.log_dt),
                      cmd_log=cmd_log.cpu().numpy(), mode_log=mode_log.cpu().numpy(),
                      mode_names=np.array(MODE_NAMES), fell=fell.cpu().numpy(),
                      fall_time=fall_time.cpu().numpy(), policy=r.policy_name, model=model_file)
        save_path = f"{save_dir}/{env_name}_data_{i+1:02d}.npz"
        rom_save_path = f"{save_dir}/{rom_env_name}_data_{i+1:02d}.npz"
        np.savez(save_path, q_log=q_np, v_log=v_np, **common)
        np.savez(rom_save_path, q_rom_log=q_to_rom(q_np), v_rom_log=v_to_rom(q_np, v_np), **common)
        print(f"  q_log {q_np.shape}, v_log {v_np.shape}")
        print(f"  Saved full-order to: {save_path}")
        print(f"  Saved ROM to:        {rom_save_path}")

    print(f"\nFinished generating {N_datasets} datasets.")
    print(f"Total rollout time: {tot_time:.1f}s")
    print(f"Total trajectories: {N_datasets * batch_size}")
