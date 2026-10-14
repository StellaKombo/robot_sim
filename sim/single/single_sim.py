##
#
#  Real-time, deterministic, single-environment closed-loop G1 simulation
#
##

# standard imports
import argparse
import math
import os
import sys
import time
import warnings
import numpy as np

# silence pygame support warning
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pygame

# mujoco sim imports
import mujoco
import mujoco.viewer

# torch imports
import torch

# for importing policy config and joystick utils
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT_DIR)
from policy.actor import load_policy
from policy.config import G1Config
from utils.joystick_utils import pygame_to_joystick_state
from utils.math_utils import gravity_from_quat

# fixed viewer render rate (Hz). Physics still runs at args.sim_dt.
RENDER_HZ = 50.0


############################################################################
# OBSERVATION / ACTION
############################################################################

def build_observation(mj_data, robot_cfg, prev_action, nu, device):
    """Build the 80-dim observation vector consumed by the policy."""

    # take mujoco state data
    omega = mj_data.sensordata[14:17].astype(np.float32)
    quat = mj_data.qpos[3:7].astype(np.float32)
    gravity = gravity_from_quat(quat)
    qpos_j = mj_data.qpos[7:7 + nu].astype(np.float32)
    qvel_j = mj_data.qvel[6:6 + nu].astype(np.float32)

    # desired twist command (vx, vy, omega)
    cmd = np.asarray(robot_cfg.cmd, dtype=np.float32)
    default_j = np.asarray(robot_cfg.default_joint_pos, dtype=np.float32)

    # robot gait phase
    phase = (mj_data.time % robot_cfg.gait_period) / robot_cfg.gait_period
    gait_phase = np.array(
        [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
        dtype=np.float32,
    )
    if np.linalg.norm(cmd) < robot_cfg.stand_cmd_threshold:
        gait_phase[:] = 0.0

    # build the observation vector
    obs = np.concatenate([
        omega,                  # 0:3
        gravity,                # 3:6
        cmd,                    # 6:9
        gait_phase,             # 9:11
        qpos_j - default_j,     # 11:34
        qvel_j,                 # 34:57
        prev_action,            # 57:80
    ]).astype(np.float32)

    return torch.from_numpy(obs).unsqueeze(0).to(device)


def compute_torque(action_np, mj_data, robot_cfg, nu):
    """PD torque from raw policy output."""
    # scale action to get desired joint positions
    qpos_des = action_np * robot_cfg.action_scale + robot_cfg.default_joint_pos

    # compute torque from PD (zero desired velocity)
    qpos_j = mj_data.qpos[7:7 + nu]
    qvel_j = mj_data.qvel[6:6 + nu]
    tau = robot_cfg.Kp * (qpos_des - qpos_j) + robot_cfg.Kd * (-qvel_j)

    return tau


############################################################################
# JOYSTICK
############################################################################

def init_joystick():
    """Initialize pygame + joystick. Returns the joystick handle or None."""
    
    # initialize pygame
    pygame.init()
    pygame.joystick.init()
    
    # look for joysticks and initialize the first one we find (if any)
    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Using default cmd from G1Config.")
        return None
    joy = pygame.joystick.Joystick(0)
    joy.init()

    print(f"Joystick connected: [{joy.get_name()}]. Driving cmd from joystick.")
    print("  Left stick:  forward/back -> vx,  left/right -> vy")
    print("  Right stick: left/right   -> yaw rate")

    return joy


def cmd_from_joystick(joy, cmd_scale):
    """Pump pygame events and return a (vx, vy, omega) cmd, or None on read error."""

    # consume the event queue (also surfaces connect/disconnect)
    pygame.event.pump()
    try:
        state = pygame_to_joystick_state(joy)
    except pygame.error:
        return None
    
    # remap the raw joystick commands
    cmd = np.array([
        state.LS_Y * cmd_scale[0],   # vx
        state.LS_X * cmd_scale[1],   # vy
        state.RS_X * cmd_scale[2],   # omega
    ], dtype=np.float32)

    return cmd


############################################################################
# MAIN
############################################################################

def main():

    # parse command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=os.path.join(ROOT_DIR, "models", "g1_23dof.xml"),
                        help="path to MuJoCo XML")
    parser.add_argument("--policy", default=os.path.join(ROOT_DIR, "policy", "g1_23dof_vel.pt"),
                        help="path to RSL-RL .pt checkpoint")
    parser.add_argument("--sim_dt", type=float, default=0.002,
                        help="physics timestep")
    parser.add_argument("--cmd", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="initial velocity command [vx, vy, omega_z] in body frame "
                             "(overridden by joystick when connected)")
    args = parser.parse_args()

    # determinism
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(False)

    # choose deive to do inference on
    device = torch.device("cpu")

    # robot + model setup
    robot_cfg = G1Config()
    nu = len(robot_cfg.Kp)

    # load MuJoCo model and data
    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_model.opt.timestep = args.sim_dt
    mj_data = mujoco.MjData(mj_model)

    # control decimation for policy queries
    control_ratio = robot_cfg.control_dt / args.sim_dt
    if not math.isclose(control_ratio, round(control_ratio), abs_tol=1e-9):
        raise ValueError(
            f"sim_dt ({args.sim_dt}) must divide control_dt "
            f"({robot_cfg.control_dt}) exactly."
        )
    decimation = int(round(control_ratio))

    # deterministic initial condition = nominal pose, zero velocity
    mj_data.qpos[:7] = robot_cfg.default_base_pos
    mj_data.qpos[7:7 + nu] = robot_cfg.default_joint_pos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # policy
    policy = load_policy(args.policy, device)
    print(f"Loaded policy from [{args.policy}].")
    print(f"sim_dt={args.sim_dt}, control_dt={robot_cfg.control_dt}, "
          f"decimation={decimation}")

    # apply CLI cmd if provided (joystick still overrides per control tick)
    if args.cmd is not None:
        robot_cfg.cmd = np.asarray(args.cmd, dtype=np.float32)
        print(f"Initial cmd: {robot_cfg.cmd.tolist()}")

    # joystick (optional — only used if connected at startup)
    joy = init_joystick()

    # mujoco viewer setup
    render_dt = 1.0 / RENDER_HZ
    print(f"Viewer render rate: {RENDER_HZ:.0f} Hz. "
          "Close the viewer window to exit.")

    # observation and action buffers
    prev_action = np.zeros(nu, dtype=np.float32)
    action = prev_action.copy()
    step = 0

    # run the sim + viewer loop. 
    try:
        with mujoco.viewer.launch_passive(
            mj_model, mj_data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            
            # set camera for better view
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = (0.0, 0.0, 0.8)

            wall_start = time.time()
            next_render = wall_start

            while viewer.is_running():
                # advance physics until sim time catches up to wall-clock time
                sim_target = min(
                    time.time() - wall_start,
                    mj_data.time + 4.0 * render_dt,
                )
                while mj_data.time < sim_target:
                    if step % decimation == 0:
                        if joy is not None:
                            new_cmd = cmd_from_joystick(joy, robot_cfg.cmd_scale)
                            if new_cmd is not None:
                                robot_cfg.cmd = new_cmd
                        obs = build_observation(mj_data, robot_cfg, prev_action, nu, device)
                        action = policy(obs).squeeze(0).cpu().numpy().astype(np.float32)
                        prev_action = action
                    mj_data.ctrl[:] = compute_torque(action, mj_data, robot_cfg, nu)
                    mujoco.mj_step(mj_model, mj_data)
                    step += 1

                # render a single frame at the fixed render rate
                viewer.sync()
                next_render += render_dt
                slack = next_render - time.time()
                if slack > 0:
                    time.sleep(slack)
                else:
                    # fell behind — resync the render clock to "now" so the
                    # deficit doesn't accumulate forever
                    next_render = time.time()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
