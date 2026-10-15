##
#
#  Controller base: shared state, command-source interface, and the twist publish loop.
#
##

# standard imports
import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass

# third-party imports
import numpy as np
import zmq

# repo root: ROBOT_SIM_ROOT_DIR from the conda env, else three levels up from this file
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)
from policy.config import G1Config
from utils.zmq_utils import CMD_PORT, connect_pub, pack_twist


############################################################################
# ROBOT STATE
############################################################################

@dataclass
class RobotState:
    """Full robot state seen by the command sources; for now a fixed nominal state, not the sim's."""
    t: float           # [s] since the controller started
    q: np.ndarray      # (nq,) base position (3), base quaternion wxyz (4), joint positions
    v: np.ndarray      # (nv,) base linear velocity (3), base angular velocity (3), joint velocities


def nominal_state(cfg: G1Config, t: float) -> RobotState:
    """Default standing pose with zero velocity everywhere."""
    # position: default base pose then default joints; velocity: all zero
    q = np.concatenate([cfg.default_base_pos, cfg.default_joint_pos]).astype(np.float64)
    v = np.zeros(6 + len(cfg.default_joint_pos))
    return RobotState(t=t, q=q, v=v)


############################################################################
# COMMAND SOURCE
############################################################################

class CommandSource:
    """Base for every controller: return the body-frame [vx, vy, wz] for a RobotState.

    `get` is called every control_dt; the sim clips the result to cmd_scale.
    """

    def __init__(self, cmd_scale: np.ndarray):
        self.cmd_scale = np.asarray(cmd_scale, dtype=np.float32)

    def get(self, state: RobotState) -> np.ndarray:
        raise NotImplementedError

    def close(self):
        pass


def import_pygame():
    """pygame without its banner and import warnings; only the sources that need it call this."""
    # hide the banner and warnings pygame prints on import
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pygame
    return pygame


############################################################################
# NODE
############################################################################

def run(cmd_source: CommandSource, cfg: G1Config):
    """Publish a twist command every control_dt of wall time, until Ctrl+C."""
    # connect to the sim's command socket
    ctx = zmq.Context.instance()
    cmd_pub = connect_pub(ctx, CMD_PORT)
    period = cfg.control_dt
    print(f"Publishing commands to simulation.py on port {CMD_PORT} at {1.0 / period:.0f} Hz.")

    # controller clock starts now
    t0 = time.perf_counter()
    next_tick = t0
    try:
        while True:
            # ask the source for a command and send it
            state = nominal_state(cfg, time.perf_counter() - t0)
            cmd_pub.send(pack_twist(cmd_source.get(state)))

            # fixed-rate ticks; after a stall, resync instead of firing a burst of late ticks
            next_tick += period
            slack = next_tick - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nController stopped.")
    finally:
        cmd_pub.close()


def main(source_class: type, description: str):
    """Entry point for each control_*.py: build the source and run until Ctrl+C."""
    # no flags; parsing still gives --help and rejects typos
    parser = argparse.ArgumentParser(description=f"{description} (pair with simulation.py).")
    parser.parse_args()

    # build the source and publish until Ctrl+C
    cfg = G1Config()
    cmd_source = source_class(cfg.cmd_scale)
    try:
        run(cmd_source, cfg)
    finally:
        cmd_source.close()
