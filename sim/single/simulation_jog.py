##
#
#  Single-robot simulation of the G1 29-DoF JOG policy (see simulation_base.py).
#  Controllers send twists up to the walk cmd_scale; by default those are read as stick
#  positions and fitted into the jog's trained bands, as deploy_robot's jog controller does.
#
##

# standard imports
import argparse
import math
import os
import sys

# third-party imports
import numpy as np
import torch

# repo root: ROBOT_SIM_ROOT_DIR from the conda env, else three levels up from this file
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy.jog.jog_actor import JogActor
from policy.walk.config import G1Config
from utils.math_utils import gravity_from_quat
from simulation_base import SimulationBase, SIM_DT, LOG_HZ

# GLOBAL VARIABLES
XML_PATH = os.path.join(ROOT_DIR, "models", "g1_29dof", "g1_29dof.xml")
POLICY_PATH = os.path.join(ROOT_DIR, "policy", "jog", "g1_29dof_jog_lstm_v0_cont.onnx")

# trained twist bands (deploy_robot g1_29dof_jog_unicycle_lstm.yaml), same as parallel_sim_jog.py
STICK_DEADBAND = 0.15
FWD_VX = (0.50, 1.50)
BWD_VX = (-1.00, -0.50)
CURVE_WZ = (-0.50, 0.50)
PIVOT_WZ = (1.00, 2.00)
RAW_LIMITS = np.array([[-1.0, 0.0, -2.0], [1.5, 0.0, 2.0]], dtype=np.float32)   # policy's twist range
MODE_NAMES = ["stand", "forward", "backward", "pivot"]   # same indices as parallel_sim_jog.py


# |stick| in [deadband, 1] -> [lo, hi] linearly (deploy_robot utils/locomotion/jog_modes.py)
def remap_stick(stick, lo, hi):
    t = (min(abs(float(stick)), 1.0) - STICK_DEADBAND) / (1.0 - STICK_DEADBAND)
    return lo + (hi - lo) * float(np.clip(t, 0.0, 1.0))


# sticks in [-1, 1] -> (mode index, twist); drive stick has priority, parked drive + steer = pivot
def shape_jog_twist(drive, steer):
    if abs(drive) >= STICK_DEADBAND:
        fwd = drive > 0.0
        vx = remap_stick(drive, *FWD_VX) if fwd else remap_stick(drive, BWD_VX[1], BWD_VX[0])
        wz = remap_stick(steer, 0.0, CURVE_WZ[1] if steer >= 0.0 else CURVE_WZ[0])
        return (1 if fwd else 2), np.array([vx, 0.0, wz], dtype=np.float32)
    if abs(steer) >= STICK_DEADBAND:
        mag = remap_stick(steer, *PIVOT_WZ)
        return 3, np.array([0.0, 0.0, mag if steer >= 0.0 else -mag], dtype=np.float32)
    return 0, np.zeros(3, dtype=np.float32)


############################################################################
# SIMULATION
############################################################################

class JogSimulation(SimulationBase):
    """G1 29-DoF running the jog policy (ONNX via onnxruntime), with its 43-frame gait clock."""

    full_log_name = "g1_29dof_jog_data.npz"
    rom_log_name = "g1_rom_jog_data.npz"

    def __init__(self, xml_path: str, policy_path: str, sim_dt: float, raw_cmd: bool = False,
                 headless: bool = False, log_name: str = None, log_hz: float = LOG_HZ):
        # policy and its deployment params (from the ONNX metadata)
        self.policy = JogActor(policy_path)
        self.policy_name = os.path.basename(policy_path)
        meta = self.policy.meta
        self.control_dt = 0.02
        self.Kp = np.asarray(meta["joint_stiffness"])
        self.Kd = np.asarray(meta["joint_damping"])
        self.action_scale = np.asarray(meta["action_scale"])
        self.default_joint_pos = np.asarray(meta["default_joint_pos"])
        self.default_base_pos = np.array([0.0, 0.0, 0.776, 1.0, 0.0, 0.0, 0.0])
        self.default_j = self.default_joint_pos.astype(np.float32)
        self.period = int(meta["motion_period_frames"])

        # controllers' full-scale twist (the walk cmd_scale) -> stick = cmd / ctrl_scale
        self.raw_cmd = raw_cmd
        self.ctrl_scale = np.asarray(G1Config().cmd_scale, dtype=np.float32)
        super().__init__(xml_path, sim_dt, headless=headless, log_name=log_name, log_hz=log_hz)

        # pelvis IMU orientation, for projected gravity (what deploy_robot and parallel_sim_jog use)
        quat_adr = self.model.sensor("pelvis_imu_quat_sensor").adr[0]
        self.quat_slice = slice(quat_adr, quat_adr + 4)

        # the actuators must be the policy's joints, in its order
        act_joints = [self.model.joint(self.model.actuator_trnid[i, 0]).name for i in range(self.nu)]
        assert act_joints == list(meta["joint_names"]), "actuator order differs from the policy's joint order"
        print(f"Loaded {self.policy.kind} jog policy [{self.policy_name}], gait period {self.period} frames; "
              + ("commands taken as raw jog twists." if raw_cmd else
                 "commands fitted into the jog bands (forward 0.5-1.5, backward -0.5 to -1.0, "
                 "pivot |wz| 1-2, else stand)."))

    def reset_policy(self):
        self.policy.reset(1)
        self.phase_step = 0
        self.mode = 0
        self.log_mode = []

    # COMMAND

    def set_cmd(self, cmd: np.ndarray):
        """Fit the controller's twist into the jog bands (or clip it, with raw_cmd); vy is dropped."""
        cmd = np.asarray(cmd, dtype=np.float32)
        if self.raw_cmd:
            self.cmd = np.clip(cmd, RAW_LIMITS[0], RAW_LIMITS[1])
            self.mode = (0 if not self.cmd.any() else 1 if self.cmd[0] > 0
                         else 2 if self.cmd[0] < 0 else 3)
        else:
            drive, steer = cmd[0] / self.ctrl_scale[0], cmd[2] / self.ctrl_scale[2]
            self.mode, self.cmd = shape_jog_twist(drive, steer)

    # POLICY

    def build_observation(self) -> np.ndarray:
        """98-dim jog observation, in the policy's observation_names order."""
        d = self.data
        omega = d.sensordata[self.gyro_slice].astype(np.float32)
        qpos_j = d.qpos[7:7 + self.nu].astype(np.float32)
        qvel_j = d.qvel[6:6 + self.nu].astype(np.float32)

        # gait clock (sin, cos) of 2*pi*t/T, blanked to (0, 0) on a zero command (idle stop)
        if self.cmd.any():
            ang = 2.0 * math.pi * self.phase_step / self.period
            phase = np.array([math.sin(ang), math.cos(ang)], dtype=np.float32)
        else:
            phase = np.zeros(2, dtype=np.float32)

        return np.concatenate([
            omega,                                          # base_ang_vel (3)
            qpos_j - self.default_j,                        # joint_pos (29)
            qvel_j,                                         # joint_vel (29)
            self.prev_action,                               # actions (29)
            self.cmd,                                       # commanded_twist (3)
            phase,                                          # motion_phase (2)
            gravity_from_quat(d.sensordata[self.quat_slice].astype(np.float32)),   # projected_gravity (3)
        ]).astype(np.float32)

    def compute_action(self) -> np.ndarray:
        obs = torch.from_numpy(self.build_observation()).unsqueeze(0)
        action = self.policy(obs).squeeze(0).numpy()
        self.phase_step = (self.phase_step + 1) % self.period   # free-running, as in deploy
        return action

    # LOGGING

    def _record(self):
        super()._record()
        self.log_mode.append(self.mode)

    def log_extras(self) -> dict:
        return dict(mode_log=np.asarray(self.log_mode, dtype=np.int8)[None],
                    mode_names=np.array(MODE_NAMES), policy=self.policy_name,
                    fell=np.array([bool(np.min(np.stack(self.log_q)[:, 2]) < 0.45)]))


############################################################################
# MAIN
############################################################################

def main():

    # parse command-line args
    parser = argparse.ArgumentParser(description="Single-robot G1 29-DoF jog simulation, driven by a control_*.py.")
    parser.add_argument("--log", default=None, metavar="NAME",
                        help="save the run to data/data/NAME/ when the sim exits")
    parser.add_argument("--log_hz", type=float, default=LOG_HZ, metavar="HZ",
                        help=f"logging rate; must divide the {1.0 / SIM_DT:g} Hz sim rate (default {LOG_HZ:g})")
    parser.add_argument("--raw_cmd", action="store_true",
                        help="take commands as real jog twists (only clipped) instead of fitting them into the jog bands; "
                             "REQUIRED for control_custom_jog.py, leave off for control_kb.py / control_joy.py")
    args = parser.parse_args()

    # build the sim and run until the viewer closes or Ctrl+C
    sim = JogSimulation(XML_PATH, POLICY_PATH, SIM_DT, raw_cmd=args.raw_cmd,
                        log_name=args.log, log_hz=args.log_hz)
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # save even after Ctrl+C
        if args.log is not None:
            sim.save_log()


if __name__ == "__main__":
    main()
