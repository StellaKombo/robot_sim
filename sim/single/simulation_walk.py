##
#
#  Single-robot simulation of the G1 23-DoF WALK velocity policy (see simulation_base.py).
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
from policy.walk.walk_actor import load_policy
from policy.walk.config import G1Config
from utils.math_utils import gravity_from_quat
from simulation_base import SimulationBase, SIM_DT, LOG_HZ

# GLOBAL VARIABLES
XML_PATH = os.path.join(ROOT_DIR, "models", "g1_23dof", "g1_23dof.xml")
POLICY_PATH = os.path.join(ROOT_DIR, "policy", "walk", "g1_23dof_vel.pt")


############################################################################
# SIMULATION
############################################################################

class WalkSimulation(SimulationBase):
    """G1 23-DoF running the walk velocity policy; commands are clipped to cmd_scale."""

    full_log_name = "g1_23dof_data.npz"
    rom_log_name = "g1_rom_data.npz"

    def __init__(self, xml_path: str, policy_path: str, sim_dt: float, robot_cfg: G1Config = None,
                 headless: bool = False, log_name: str = None, log_hz: float = LOG_HZ):
        # robot config: gains, nominal pose, timing
        self.cfg = robot_cfg or G1Config()
        self.control_dt = self.cfg.control_dt
        self.Kp = np.asarray(self.cfg.Kp)
        self.Kd = np.asarray(self.cfg.Kd)
        self.action_scale = np.asarray(self.cfg.action_scale)
        self.default_base_pos = np.asarray(self.cfg.default_base_pos)
        self.default_joint_pos = np.asarray(self.cfg.default_joint_pos)
        self.default_j = self.default_joint_pos.astype(np.float32)
        self.cmd_scale = np.asarray(self.cfg.cmd_scale, dtype=np.float32)

        # policy
        self.device = torch.device("cpu")
        self.policy = load_policy(policy_path, self.device)
        super().__init__(xml_path, sim_dt, headless=headless, log_name=log_name, log_hz=log_hz)

    # COMMAND

    def set_cmd(self, cmd: np.ndarray):
        """Body-frame [vx, vy, wz], clipped to the policy's training range."""
        self.cmd = np.clip(np.asarray(cmd, dtype=np.float32), -self.cmd_scale, self.cmd_scale)

    # POLICY

    def build_observation(self) -> torch.Tensor:
        """80-dim policy observation, in the exact layout the policy was trained on."""

        d = self.data

        # sensor readings, cast to float32 as in training
        omega = d.sensordata[self.gyro_slice].astype(np.float32)
        gravity = gravity_from_quat(d.qpos[3:7].astype(np.float32))
        qpos_j = d.qpos[7:7 + self.nu].astype(np.float32)
        qvel_j = d.qvel[6:6 + self.nu].astype(np.float32)

        # gait clock from sim time, zeroed below stand_cmd_threshold
        phase = (d.time % self.cfg.gait_period) / self.cfg.gait_period
        gait_phase = np.array([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
                              dtype=np.float32)
        if np.linalg.norm(self.cmd) < self.cfg.stand_cmd_threshold:
            gait_phase[:] = 0.0

        # assemble in the order the policy was trained on
        obs = np.concatenate([
            omega,                      # 0:3
            gravity,                    # 3:6
            self.cmd,                   # 6:9
            gait_phase,                 # 9:11
            qpos_j - self.default_j,    # 11:34
            qvel_j,                     # 34:57
            self.prev_action,           # 57:80
        ]).astype(np.float32)
        return torch.from_numpy(obs).unsqueeze(0).to(self.device)

    def compute_action(self) -> np.ndarray:
        return self.policy(self.build_observation()).squeeze(0).cpu().numpy()


############################################################################
# MAIN
############################################################################

def main():

    # parse command-line args
    parser = argparse.ArgumentParser(description="Single-robot G1 23-DoF walk simulation, driven by a control_*.py.")
    parser.add_argument("--log", default=None, metavar="NAME",
                        help="save the run to data/data/NAME/ when the sim exits")
    parser.add_argument("--log_hz", type=float, default=LOG_HZ, metavar="HZ",
                        help=f"logging rate; must divide the {1.0 / SIM_DT:g} Hz sim rate (default {LOG_HZ:g})")
    args = parser.parse_args()

    # build the sim and run until the viewer closes or Ctrl+C
    sim = WalkSimulation(XML_PATH, POLICY_PATH, SIM_DT, G1Config(), log_name=args.log, log_hz=args.log_hz)
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
