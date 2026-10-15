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
from utils.zmq_utils import CMD_PORT, STATE_PORT, connect_pub, connect_sub, pack_twist, unpack_state


############################################################################
# ROBOT STATE
############################################################################

@dataclass
class RobotState:
    """Full robot state from simulation.py, as of its latest policy tick."""
    t: float           # [s] sim time
    q: np.ndarray      # (nq,) base position (3), base quaternion wxyz (4), joint positions
    v: np.ndarray      # (nv,) base linear vel, world frame (3), base angular vel, body frame (3), joint vels


def nominal_state(cfg: G1Config) -> RobotState:
    """Default standing pose with zero velocity everywhere."""
    # position: default base pose then default joints; velocity: all zero
    q = np.concatenate([cfg.default_base_pos, cfg.default_joint_pos]).astype(np.float64)
    v = np.zeros(6 + len(cfg.default_joint_pos))
    return RobotState(t=0.0, q=q, v=v)


############################################################################
# COMMAND SOURCE
############################################################################

class CommandSource:
    """Base for every controller: return the body-frame [vx, vy, wz] for a RobotState.

    `get` is called every control_dt; the sim clips the result to cmd_scale. `self.t_start` is the
    sim time of the first state received, so `state.t - self.t_start` is a controller clock that
    starts at 0. A source may set `self.ref` to a world-frame [x, y] in `get`, and the sim draws it
    as a ball on the floor.
    """

    def __init__(self, cmd_scale: np.ndarray):
        self.cmd_scale = np.asarray(cmd_scale, dtype=np.float32)
        self.t_start = 0.0   # [s] sim time of the first state received, set by run()
        self.ref = None      # optional [x, y] reference point to display in the sim

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
    """Every control_dt of wall time, read the sim's newest state and publish a twist, until Ctrl+C.

    Until the sim's first state arrives, the source sees the nominal state at t = 0.
    """
    # connect to the sim's command and state sockets
    ctx = zmq.Context.instance()
    cmd_pub = connect_pub(ctx, CMD_PORT)
    state_sub = connect_sub(ctx, STATE_PORT)
    period = cfg.control_dt
    nu = len(cfg.default_joint_pos)
    print(f"Publishing commands to simulation.py on port {CMD_PORT} at {1.0 / period:.0f} Hz, "
          f"reading its state on port {STATE_PORT}.")

    state = nominal_state(cfg)
    have_state = False
    last_t = 0.0
    next_tick = time.perf_counter()
    try:
        while True:
            # newest state from the sim, if a new one arrived
            try:
                t, q, v = unpack_state(state_sub.recv(zmq.NOBLOCK), 7 + nu, 6 + nu)
                state = RobotState(t=t, q=q, v=v)

                # zero the controller clock at the first state, and again if sim time jumps back (restart)
                if not have_state or t < last_t:
                    print(f"Receiving state from simulation.py; zeroing controller time at sim t = {t:.2f} s.")
                    cmd_source.t_start = t
                    have_state = True
                last_t = t
            except zmq.Again:
                pass

            # ask the source for a command, then send it with any reference point it set
            cmd = cmd_source.get(state)
            cmd_pub.send(pack_twist(cmd, cmd_source.ref))

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
        state_sub.close()


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
