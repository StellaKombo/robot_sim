##
#
#  Plot the ROM state of randomly sampled trajectories from a _rom_data.npz dataset.
# 
##

import argparse
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# project root (for symmetry with plot_23dof_data.py — no utils currently needed)
ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)


############################################################################
# USEFUL VARS AND FUNCTIONS
############################################################################

# ROM state labels — q_rom = [px, py, theta] (world), v_rom = [vx, vy, wz] (body)
Q_ROM_LABELS = ["px [m]", "py [m]", "theta [rad]"]
V_ROM_LABELS = ["vx [m/s]", "vy [m/s]", "wz [rad/s]"]


def draw_traces(axes, t, X, labels, overlays=None):
    """Draw one line per trajectory on each axis. Optional dashed overlay."""
    for d, ax in enumerate(axes):
        lines = ax.plot(t, X[:, :, d].T, linewidth=0.8, alpha=0.8)
        if overlays is not None and d in overlays:
            for traj_i, line in enumerate(lines):
                ax.plot(t, overlays[d][traj_i], linestyle="--",
                        color=line.get_color(), linewidth=0.8, alpha=0.8)
        ax.set_title(labels[d], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]", fontsize=8)
        ax.tick_params(labelsize=7)


def playback_rom_trajectories(q_rom_sel, sim_dt, idx, render_hz=50.0,
                              tri_length=0.25, tri_width=0.18, loop=True):
    """Animate selected ROM poses [px, py, theta] in a 2D top-down view in its
    own figure. Each trajectory is drawn as a triangle pointing along its
    heading, with a trail of past positions. Paced to wall clock at render_hz.

    If loop=True the animation restarts forever; close the window to stop.
    """
    num, T, _ = q_rom_sel.shape
    render_dt = 1.0 / render_hz
    step_skip = max(1, int(round(render_dt / sim_dt)))

    # triangle template in body frame (nose along +x)
    triangle_body = np.array([
        [ tri_length,        0.0 ],
        [-tri_length / 2,  tri_width / 2 ],
        [-tri_length / 2, -tri_width / 2 ],
    ])  # (3, 2)

    # figure + axes setup
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("px [m]")
    ax.set_ylabel("py [m]")
    ax.set_title(f"ROM playback ({num} traj){' — looping' if loop else ''}")

    pad = max(0.5, tri_length * 4)
    ax.set_xlim(q_rom_sel[..., 0].min() - pad, q_rom_sel[..., 0].max() + pad)
    ax.set_ylim(q_rom_sel[..., 1].min() - pad, q_rom_sel[..., 1].max() + pad)

    # one triangle + one trail line per trajectory; default cycle matches the
    # color order used by draw_traces() so an env_idx looks the same in both views
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    triangles = []
    trails = []
    for traj_i in range(num):
        color = color_cycle[traj_i % len(color_cycle)]
        tri = Polygon(triangle_body.tolist(), closed=True, facecolor=color,
                      edgecolor="black", alpha=0.85, zorder=3,
                      label=f"env {idx[traj_i]}")
        ax.add_patch(tri)
        trail, = ax.plot([], [], color=color, linewidth=1.0, alpha=0.6)
        triangles.append(tri)
        trails.append(trail)
    ax.legend(loc="best", fontsize=8)

    print(f"\nPlaying back {num} ROM trajectory(ies). Close the window to {'stop the loop' if loop else 'abort'}.")

    plt.ion()
    plt.show(block=False)
    fig.canvas.draw()
    fig_num = getattr(fig, "number", None)

    while True:
        # reset trails at the start of each pass
        for trail in trails:
            trail.set_data([], [])
        wall_start = time.time()

        for k in range(0, T, step_skip):
            if fig_num is None or not plt.fignum_exists(fig_num):
                plt.ioff()
                return
            # pace playback to real time
            target = wall_start + k * sim_dt
            slack = target - time.time()
            if slack > 0:
                time.sleep(slack)

            for traj_i in range(num):
                px, py, theta = q_rom_sel[traj_i, k]
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                verts = triangle_body @ R.T + np.array([px, py])
                triangles[traj_i].set_xy(verts)
                trails[traj_i].set_data(q_rom_sel[traj_i, :k + 1, 0],
                                        q_rom_sel[traj_i, :k + 1, 1])

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        if not loop:
            break

    plt.ioff()


############################################################################
# MAIN
############################################################################

def main():

    # which file: a simulation.py run from data/data/<run>/, else parallel_sim's dataset _01
    parser = argparse.ArgumentParser(description="Plot a saved run or dataset.")
    parser.add_argument("run", nargs="?", default=None,
                        help="run name given to simulation.py --log; omit for data/data/g1_rom_data_01.npz")
    args = parser.parse_args()
    if args.run is None:
        data_path = os.path.join(ROOT_DIR, "data", "data", "g1_rom_data_01.npz")
    else:
        data_path = os.path.join(ROOT_DIR, "data", "data", args.run, "g1_rom_data.npz")

    # trajectories to plot
    num_trajectories = 3

    # choose the random seed to select the same trajectories
    # seed = 0
    seed = int(time.time())

    # load the data
    data = np.load(data_path)
    q_rom = data["q_rom_log"]            # (B, T, 3) [px, py, theta]
    v_rom = data["v_rom_log"]            # (B, T, 3) [vx, vy, wz]  (body frame)
    cmd = data["cmd_log"]                # (B, T, 3) [vx, vy, wz]  (body frame)
    sim_dt = float(data["sim_dt"])
    control_dt = float(data["control_dt"])
    B, T, _ = q_rom.shape
    print(f"loaded: {data_path}")
    print(f"  B={B}, T={T}")
    print(f"  sim_dt={sim_dt}, control_dt={control_dt}")

    # sample trajectories to plot
    num = min(num_trajectories, B)
    rng = np.random.default_rng(seed)
    idx = rng.choice(B, size=num, replace=False)
    idx.sort()
    print(f"selected {num} trajectories: {idx.tolist()}")

    # select the lucky trajectories
    q_rom_sel = q_rom[idx]              # (num, T, 3)
    v_rom_sel = v_rom[idx]              # (num, T, 3)
    cmd_sel = cmd[idx]                  # (num, T, 3)
    t = np.arange(T) * sim_dt           # (T,)

    # v_rom and cmd are both in the body frame
    v_overlays = {
        0: cmd_sel[:, :, 0],
        1: cmd_sel[:, :, 1],
        2: cmd_sel[:, :, 2],
    }

    # Figure 2 — static traces: q_rom on the top row, v_rom on the bottom row
    fig_static, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    draw_traces(axes[0], t, q_rom_sel, Q_ROM_LABELS)
    draw_traces(axes[1], t, v_rom_sel, V_ROM_LABELS, overlays=v_overlays)
    fig_static.suptitle("ROM traces — q_rom (top), v_rom (bottom, dashed = cmd)",
                        fontsize=11)

    # Figure 1 — looping 2D top-down animation (blocks until the window is closed)
    playback_rom_trajectories(q_rom_sel, sim_dt, idx, loop=True)

    # keep static figure visible after the animation window is closed
    plt.show()


if __name__ == "__main__":
    main()
