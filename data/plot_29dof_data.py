##
#
#  Plot the full state of randomly sampled trajectories from a g1_29dof_jog_data_XX.npz dataset,
#  then replay them in MuJoCo. 29-DoF counterpart of plot_23dof_data.py, reusing its helpers.
#
##

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT_DIR = os.getenv("ROBOT_SIM_ROOT_DIR",
                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "data"))
from utils.math_utils import body_to_world


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot and replay a 29-DoF jog dataset or run.")
    parser.add_argument("run", nargs="?", default=None,
                        help="run name given to simulation.py --log; omit for a GPU dataset")
    parser.add_argument("--dataset", type=int, default=1, help="GPU dataset number XX (default 1)")
    parser.add_argument("--num", type=int, default=3, help="trajectories to show (default 3)")
    parser.add_argument("--seed", type=int, default=None, help="selection seed (default: time)")
    parser.add_argument("--save", nargs="?", const="", default=None, metavar="DIR",
                        help="save position and velocity PNGs and exit; optional output directory "
                             "(default: <dataset>_plots beside the NPZ file)")
    args = parser.parse_args()
    if args.run is None:
        data_path = os.path.join(ROOT_DIR, "data", "data", f"g1_29dof_jog_data_{args.dataset:02d}.npz")
    else:
        data_path = os.path.join(ROOT_DIR, "data", "data", args.run, "g1_29dof_jog_data.npz")
    xml_path = os.path.join(ROOT_DIR, "models", "g1_29dof", "g1_29dof.xml")

    if not os.path.isfile(data_path):
        log_dir = Path(ROOT_DIR) / "data" / "data"
        runs = sorted(path.parent.name for path in log_dir.glob("*/g1_29dof_jog_data.npz"))
        hint = ("Available saved runs: " + ", ".join(runs) + ".\n"
                f"Plot one with: python data/plot_29dof_data.py {runs[0]}"
                if runs else "Record a run with simulation.py --log <name> first.")
        parser.error(f"Jogging dataset not found: {data_path}\n{hint}")

    # Select a non-interactive backend before either pyplot or the shared
    # plotting helpers are imported. Saving never launches the GLFW viewer.
    if args.save is not None:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_23dof_data import state_labels, plot_data, playback_trajectories

    save_dir = None
    if args.save is not None:
        save_dir = (Path(args.save).expanduser() if args.save else
                    Path(data_path).with_name(Path(data_path).stem + "_plots"))
        save_dir.mkdir(parents=True, exist_ok=True)

    # load the data
    data = np.load(data_path, allow_pickle=False)
    q, v, cmd = data["q_log"], data["v_log"], data["cmd_log"]   # (B,T,nq), (B,T,nv), (B,T,3)
    log_dt = float(data["log_dt"])
    B, T, nq = q.shape
    print(f"loaded: {data_path}")
    print(f"  B={B}, T={T}, nq={nq}, nv={v.shape[2]}, log_dt={log_dt}, policy={data['policy']}")
    print(f"  fell: {int(data['fell'].sum())}/{B}")

    # sample trajectories
    rng = np.random.default_rng(int(time.time()) if args.seed is None else args.seed)
    idx = np.sort(rng.choice(B, size=min(args.num, B), replace=False))
    print(f"selected {len(idx)} trajectories: {idx.tolist()}")
    q_sel, v_sel, cmd_sel = q[idx], v[idx], cmd[idx]
    t = np.arange(T) * log_dt

    # cmd is body frame; rotate it to world to overlay on the base velocities (vx, vy, wz)
    zero = np.zeros_like(cmd_sel[..., 0:1])
    cmd_lin = body_to_world(q_sel[..., 3:7], np.concatenate([cmd_sel[..., 0:2], zero], axis=-1))
    v_overlays = {0: cmd_lin[..., 0], 1: cmd_lin[..., 1], 5: cmd_sel[..., 2]}

    # Save static traces, or show them alongside the looping MuJoCo replay.
    q_labels, v_labels = state_labels(xml_path)
    plot_data(t, q_sel, q_labels, "positions, q",
              save_path=save_dir / "positions.png" if save_dir is not None else None)
    plot_data(t, v_sel, v_labels, "velocities, v", overlays=v_overlays,
              save_path=save_dir / "velocities.png" if save_dir is not None else None)
    if save_dir is not None:
        plt.close("all")
    else:
        plt.ion()
        plt.show(block=False)
        plt.pause(0.1)
        playback_trajectories(xml_path, q_sel, v_sel, log_dt, idx, loop=True)
        plt.ioff()
        plt.show()
