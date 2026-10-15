##
#
#  Plot the full state of randomly sampled trajectories from a _23dof_data.npz dataset.
#
##

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import mujoco
import mujoco.viewer

# project root so we can import utils.math_utils
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
from utils.math_utils import body_to_world


############################################################################
# USEFUL VARS AND FUNCTIONS
############################################################################

# floating-base state labels; the joint labels come from the model's joint names
BASE_Q_LABELS = ["px [m]", "py [m]", "pz [m]", "qw", "qx", "qy", "qz"]
BASE_V_LABELS = ["vx [m/s]", "vy [m/s]", "vz [m/s]", "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"]


def state_labels(xml_path):
    """Subplot titles for q and v: the floating base, then each hinge joint by name, in qpos order."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    names = [model.joint(j).name.removesuffix("_joint") for j in range(model.njnt)
             if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    return (BASE_Q_LABELS + [f"{n} [rad]" for n in names],
            BASE_V_LABELS + [f"{n} [rad/s]" for n in names])


def plot_data(t, X, labels, title, n_cols=5, overlays=None, save_path=None):
    """Plot every state dimension as its own subplot, with one line per trajectory.

    overlays: optional dict {subplot_index -> (num, T) array} drawn as dashed
              black lines, with a legend for the line styles.
    """
    n_dim = X.shape[2]
    n_rows = (n_dim + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 1.8 * n_rows),
                             sharex=True, constrained_layout=True)
    axes = axes.flatten()

    for d in range(n_dim):
        ax = axes[d]
        # X[traj, time, dim] -> line per trajectory
        ax.plot(t, X[:, :, d].T, linewidth=0.8, alpha=0.8)
        # optional overlay: dashed black line per trajectory
        if overlays is not None and d in overlays:
            ax.plot(t, overlays[d].T, linestyle="--", color="black", linewidth=0.8, alpha=0.8)
        ax.set_title(labels[d], fontsize=9)
        ax.grid(True, alpha=0.3)
        if d // n_cols == n_rows - 1:
            ax.set_xlabel("time [s]", fontsize=8)
        ax.tick_params(labelsize=7)

    # hide unused subplots
    for d in range(n_dim, len(axes)):
        axes[d].axis("off")

    fig.suptitle(title, fontsize=11)

    # line styles are shared across trajectories, so the legend uses neutral proxies
    if overlays is not None:
        style_key = [Line2D([], [], color="gray", linewidth=0.8, label="actual"),
                     Line2D([], [], color="black", linewidth=0.8, linestyle="--", label="commanded")]
        fig.legend(handles=style_key, loc="outside upper right", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"saved: {save_path}")


def playback_trajectories(xml_path, q_sel, v_sel, log_dt, idx, render_hz=50.0,
                          loop=True):
    """Replay selected (q, v) trajectories in a passive MuJoCo viewer, in sequence.

    Drives mj_data.qpos / qvel from the logs (no physics integration) and calls
    mj_forward to update kinematics + viewer. Real-time wall-clock paced.

    If loop=True, cycles through all trajectories forever; close the viewer
    window to stop.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    num, T, _ = q_sel.shape
    render_dt = 1.0 / render_hz
    step_skip = max(1, int(round(render_dt / log_dt)))  # render every Nth log sample

    print(f"\nPlaying back {num} trajectory(ies). "
          f"Close the window to {'stop the loop' if loop else 'abort'}.")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        # camera angle matches simulation.py
        viewer.cam.azimuth = 140
        viewer.cam.elevation = -20
        viewer.cam.distance = 4.0
        viewer.cam.lookat[:] = (0.0, 0.0, 0.8)

        while True:
            for traj_i in range(num):
                print(f"  trajectory env_idx={idx[traj_i]} ({traj_i + 1}/{num})")
                wall_start = time.time()
                for k in range(0, T, step_skip):
                    if not viewer.is_running():
                        return

                    target = wall_start + k * log_dt
                    slack = target - time.time()
                    if slack > 0:
                        time.sleep(slack)

                    data.qpos[:] = q_sel[traj_i, k]
                    data.qvel[:] = v_sel[traj_i, k]
                    mujoco.mj_forward(model, data)
                    viewer.sync()
            if not loop:
                break


############################################################################
# MAIN
############################################################################

def main():

    # which file: a simulation.py run from data/data/<run>/, else parallel_sim's dataset _01
    parser = argparse.ArgumentParser(description="Plot a saved run or dataset.")
    parser.add_argument("run", nargs="?", default=None,
                        help="run name given to simulation.py --log; omit for data/data/g1_23dof_data_01.npz")
    args = parser.parse_args()
    if args.run is None:
        data_path = os.path.join(ROOT_DIR, "data", "data", "g1_23dof_data_01.npz")
    else:
        data_path = os.path.join(ROOT_DIR, "data", "data", args.run, "g1_23dof_data.npz")
    xml_path = os.path.join(ROOT_DIR, "models", "g1_23dof.xml")

    # trajectories to plot
    num_trajectories = 3

    # choose the random seed to select the same trajectories
    # seed = 0
    seed = int(time.time())

    # load the data
    data = np.load(data_path)
    q = data["q_log"]                    # (B, T, nq)
    v = data["v_log"]                    # (B, T, nv)
    cmd = data["cmd_log"]                # (B, T, 3)
    sim_dt = float(data["sim_dt"])
    control_dt = float(data["control_dt"])
    log_dt = float(data["log_dt"]) if "log_dt" in data.files else sim_dt   # older files logged every sim step
    B, T, nq = q.shape
    nv = v.shape[2]
    print(f"loaded: {data_path}")
    print(f"  B={B}, T={T}, nq={nq}, nv={nv}")
    print(f"  sim_dt={sim_dt}, control_dt={control_dt}, log_dt={log_dt}")

    # sample trajectories to plot
    num = min(num_trajectories, B)
    rng = np.random.default_rng(seed)
    idx = rng.choice(B, size=num, replace=False)
    idx.sort()
    print(f"selected {num} trajectories: {idx.tolist()}")

    # select the lucky trajectories and plot
    q_sel = q[idx]                    # (num, T, nq)
    v_sel = v[idx]                    # (num, T, nv)
    cmd_sel = cmd[idx]                # (num, T, 3)
    t = np.arange(T) * log_dt         # (T,)

    # cmd is in the body frame
    quat_sel = q_sel[:, :, 3:7]                                              # (num, T, 4) wxyz
    zero_col = np.zeros_like(cmd_sel[..., 0:1])
    cmd_lin_body = np.concatenate([cmd_sel[..., 0:2], zero_col], axis=-1)    # (num, T, 3) (vx, vy, 0)
    cmd_ang_body = np.concatenate([zero_col, zero_col, cmd_sel[..., 2:3]], axis=-1)  # (num, T, 3) (0, 0, wz)
    cmd_lin_world = body_to_world(quat_sel, cmd_lin_body)                    # (num, T, 3)
    cmd_ang_world = body_to_world(quat_sel, cmd_ang_body)                    # (num, T, 3)

    # overlay world-frame cmd on the matching base-velocity subplots:
    v_overlays = {
        0: cmd_lin_world[:, :, 0],
        1: cmd_lin_world[:, :, 1],
        5: cmd_ang_world[:, :, 2],
    }

    # Figures 1 & 2 — static traces (created, then shown non-blocking so they
    # appear before the MuJoCo window opens)
    q_labels, v_labels = state_labels(xml_path)
    plot_data(t, q_sel, q_labels, f"positions, q")
    plot_data(t, v_sel, v_labels, f"velocities, v", overlays=v_overlays)
    plt.ion()
    plt.show(block=False)
    plt.pause(0.1)  # nudge the backend to actually paint the windows

    # Figure 3 — looping MuJoCo playback (blocks until the viewer is closed)
    playback_trajectories(xml_path, q_sel, v_sel, log_dt, idx, loop=True)

    # keep static figures visible after the MuJoCo viewer is closed
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
