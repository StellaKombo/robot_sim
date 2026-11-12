"""Plot and replay two saved ROM runs on one simulated-time clock.

No controller or physics is run. At --speed 1 (the default), 60 simulated
seconds play in 60 wall-clock seconds, independently of MPC solve times.
"""

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time
import warnings

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# The parent folder is a script collection, not a package: KCF_MPC/data.py
# already owns the top-level `data` import in training code. Do not shadow it.
PLOT_DIR = str(Path(__file__).resolve().parent)
if PLOT_DIR not in sys.path:
    sys.path.insert(0, PLOT_DIR)

from plot_rom_data import Q_ROM_LABELS, V_ROM_LABELS, pose_sincos, triangle_vertices
from utils.kcf_reference import PlanarReference

COLORS = ("#0b30c6", "#c30d0d")


@dataclass(frozen=True)
class ROMRun:
    label: str
    times: np.ndarray
    pose: np.ndarray  # [world x, world y, sin(theta), cos(theta)]
    velocity: np.ndarray  # body [vx, vy, wz]
    command: np.ndarray  # body [vx, vy, wz]
    settings: dict | None = None
    provenance: dict | None = None

    def reference(self, times):
        if self.settings is None:
            return None, None
        path = PlanarReference(self.settings)
        q, v = path.reference(times)
        tracking = path.track(q, v)
        return tracking[..., :4], tracking[..., 4:7]


def load_rom_run(run, *, env_index=0, label=None):
    """Accept a run name, run directory or NPZ; never guess current references."""
    path = Path(run).expanduser()
    if not path.exists():
        path = ROOT / "data/data" / path
    if path.is_dir():
        path = path / "g1_rom_data.npz"
    with np.load(path, allow_pickle=False) as data:
        pose = pose_sincos(data["q_rom_log"])
        if pose.ndim != 3 or pose.shape[1] < 2:
            raise ValueError(f"{path}: expected at least two samples in (B,T,3/4) poses")
        if not 0 <= env_index < pose.shape[0]:
            raise ValueError(f"{path}: env index must be in [0, {pose.shape[0] - 1}]")
        velocity, command = data["v_rom_log"], data["cmd_log"]
        expected = (*pose.shape[:2], 3)
        if velocity.shape != expected or command.shape != expected:
            raise ValueError(f"{path}: body velocity and command must have shape {expected}")
        if not np.isfinite(velocity).all() or not np.isfinite(command).all():
            raise ValueError(f"{path}: velocity and command must be finite")
        log_dt = float(data["log_dt"] if "log_dt" in data else data["sim_dt"])
        if not np.isfinite(log_dt) or log_dt <= 0:
            raise ValueError(f"{path}: log_dt must be finite and positive")
        times = np.arange(pose.shape[1]) * log_dt
        pose, velocity, command = (array[env_index].copy() for array in (pose, velocity, command))
    plan = path.parent / "trial_plan.json"
    settings = json.loads(plan.read_text())["mpc"] if plan.is_file() else None
    launcher = path.parent / "launcher_result.json"
    provenance = json.loads(launcher.read_text()) if launcher.is_file() else None
    model = settings.get("prediction_model") if settings else None
    default_label = {"kcf": "KCF", "single_integrator": "Single integrator"}.get(model, path.parent.name)
    return ROMRun(label or default_label, times, pose, velocity, command, settings, provenance)


def comparison_window(runs, start=0., end=None):
    """Use physical simulation time, not sample indices or solver wall time."""
    if len(runs) != 2:
        raise ValueError("ROM comparison requires exactly two runs")
    shared_end = min(run.times[-1] for run in runs)
    end = shared_end if end is None else end
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= shared_end + 1e-9:
        raise ValueError(f"Require 0 <= start < end <= shared logged duration ({shared_end:g} s)")
    return float(start), float(min(end, shared_end))


def sample_pose(run, sim_time):
    """Interpolate sine/cosine, avoiding yaw-wrap interpolation through zero yaw."""
    pose = np.array([np.interp(sim_time, run.times, run.pose[:, i]) for i in range(4)])
    if np.hypot(*pose[2:4]) < 1e-12:
        # An exactly antipodal pair has no unique interpolated heading.
        index = np.clip(np.searchsorted(run.times, sim_time), 0, len(run.times) - 1)
        pose[2:4] = run.pose[index, 2:4]
    return pose


def replay_time(wall_start, wall_now, start, end, speed=1.):
    """Late renders jump to the current instant instead of slowing the recording."""
    return min(end, start + max(0., wall_now - wall_start) * speed)


def warn_if_unmatched(runs):
    a, b = runs
    for run in runs:
        if run.provenance and (not run.provenance.get("sources_unchanged", False)
                               or run.provenance.get("interrupted", False)
                               or run.provenance.get("simulator", {}).get("outcome") != "complete"):
            warnings.warn(f"{run.label}: launcher flags changed sources, interruption or an incomplete run; "
                          "replay is available but this is not a verified clean benchmark.", stacklevel=2)
    if not np.allclose(a.pose[0], b.pose[0], atol=1e-6, rtol=0):
        warnings.warn("Initial poses differ; playback preserves their recorded world coordinates.", stacklevel=2)
    end = min(a.times[-1], b.times[-1])
    q_a, _ = a.reference(np.linspace(0, end, 101))
    q_b, _ = b.reference(np.linspace(0, end, 101))
    if q_a is None or q_b is None:
        warnings.warn("A trial_plan.json is missing; no current/default reference will be substituted.", stacklevel=2)
    elif not np.allclose(q_a, q_b, atol=1e-6, rtol=0):
        warnings.warn("Saved references differ; each panel shows its own recorded task.", stacklevel=2)


def plot_rom_comparison(runs, *, start=0., end=None):
    """Physical pose sine/cosine and body twist, with saved task/command overlays."""
    start, end = comparison_window(runs, start, end)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for run, color in zip(runs, COLORS):
        selected = (run.times >= start) & (run.times <= end + 1e-9)
        t = run.times[selected]
        q, v, command = run.pose[selected], run.velocity[selected], run.command[selected]
        q_ref, v_ref = run.reference(t)
        for i, ax in enumerate(axes[0]):
            ax.plot(t, q[:, i], color=color, linewidth=1., label=run.label)
            if q_ref is not None:
                ax.plot(t, q_ref[:, i], "--", color=color, alpha=.6,
                        linewidth=.9, label=f"{run.label} reference")
        for i, ax in enumerate(axes[1, :3]):
            ax.plot(t, v[:, i], color=color, linewidth=1.)
            ax.plot(t, command[:, i], ":", color=color, alpha=.65, linewidth=.8)
            if v_ref is not None:
                ax.plot(t, v_ref[:, i], "--", color=color, alpha=.6, linewidth=.9)
        if q_ref is not None:
            axes[1, 3].plot(t, np.linalg.norm(q[:, :2] - q_ref[:, :2], axis=1), color=color)
    for ax, title in zip(axes.flat, Q_ROM_LABELS + V_ROM_LABELS + ["Position error [m]"]):
        ax.set(title=title, xlabel="Simulated time [s]", xlim=(start, end))
        ax.grid(True, alpha=.25)
    axes[0, 2].set_ylim(-1.1, 1.1)
    axes[0, 3].set_ylim(-1.1, 1.1)
    axes[0, 0].legend(fontsize=8)
    axes[1, 3].text(.03, .96, "Solid: measured\nDashed: saved reference\nDotted: applied command (body twist)",
                    transform=axes[1, 3].transAxes, va="top", fontsize=8)
    fig.suptitle("ROM comparison — model-aligned heading features; world pose / body twist")
    return fig


class ROMComparisonPlayback:
    """Reusable artists; update(t) renders both recordings at the same sim time."""

    def __init__(self, runs, *, start=0., end=None, overlay=True, speed=1.):
        self.start, self.end = comparison_window(runs, start, end)
        self.runs, self.speed = runs, speed
        count = 1 if overlay else len(runs)
        self.fig, axes = plt.subplots(1, count, figsize=(7 * count, 6), squeeze=False,
                                      constrained_layout=True)
        self.axes = axes[0]
        self.triangles, self.trails, self.references = [], [], []
        all_positions = [run.pose[:, :2] for run in runs]
        for run in runs:
            q_ref, _ = run.reference(run.times)
            if q_ref is not None:
                all_positions.append(q_ref[:, :2])
        xy = np.concatenate(all_positions)
        lower, upper = xy.min(axis=0) - .5, xy.max(axis=0) + .5
        for ax in self.axes:
            ax.set(xlabel="World x [m]", ylabel="World y [m]",
                   xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=.25)
        for i, (run, color) in enumerate(zip(runs, COLORS)):
            ax = self.axes[0 if overlay else i]
            q_ref, _ = run.reference(run.times)
            reference_marker = None
            if q_ref is not None:
                selected = (run.times >= self.start) & (run.times <= self.end + 1e-9)
                ax.plot(q_ref[selected, 0], q_ref[selected, 1], "--", color="gray", linewidth=1.)
                reference_marker = Polygon(triangle_vertices(q_ref[0], .18, .13),
                                           facecolor="none", edgecolor="gray", linewidth=1.5,
                                           label="Reference" if not overlay or i == 0 else None)
                ax.add_patch(reference_marker)
            triangle = Polygon(triangle_vertices(run.pose[0]), facecolor=color,
                               edgecolor="black", alpha=.85, label=run.label, zorder=3)
            ax.add_patch(triangle)
            trail, = ax.plot([], [], color=color, linewidth=1.1, alpha=.8)
            self.triangles.append(triangle)
            self.trails.append(trail)
            self.references.append(reference_marker)
            if not overlay:
                ax.set_title(run.label)
        for ax in self.axes:
            ax.legend(loc="best", fontsize=8)
        self.title = self.fig.suptitle("")
        self.update(self.start)

    def update(self, sim_time):
        sim_time = float(np.clip(sim_time, self.start, self.end))
        for i, run in enumerate(self.runs):
            pose = sample_pose(run, sim_time)
            self.triangles[i].set_xy(triangle_vertices(pose))
            first = np.searchsorted(run.times, self.start, side="left")
            last = np.searchsorted(run.times, sim_time, side="right")
            trail = np.vstack((run.pose[first:last, :2], pose[:2]))
            self.trails[i].set_data(trail[:, 0], trail[:, 1])
            if self.references[i] is not None:
                q_ref, _ = run.reference(sim_time)
                self.references[i].set_xy(triangle_vertices(q_ref, .18, .13))
        self.title.set_text(f"ROM comparison — t={sim_time:.2f} / {self.end:.2f} s — {self.speed:g}× playback")


def playback_rom_comparison(runs, *, start=0., end=None, speed=1., render_hz=50.,
                            loop=True, overlay=True):
    """Wall-clock-paced saved-data replay; slow drawing skips obsolete frames."""
    if not np.isfinite([speed, render_hz]).all() or speed <= 0 or render_hz <= 0:
        raise ValueError("speed and render_hz must be finite and positive")
    playback = ROMComparisonPlayback(runs, start=start, end=end, overlay=overlay, speed=speed)
    was_interactive = plt.isinteractive()
    plt.ion()
    plt.show(block=False)
    playback.fig.canvas.draw()
    try:
        while plt.fignum_exists(playback.fig.number):
            wall_start = time.perf_counter()
            while plt.fignum_exists(playback.fig.number):
                frame_start = time.perf_counter()
                current = replay_time(wall_start, frame_start, playback.start, playback.end, speed)
                playback.update(current)
                playback.fig.canvas.draw_idle()
                playback.fig.canvas.flush_events()
                if current >= playback.end:
                    break
                # Pause pumps GUI events; rendering time counts towards playback time.
                slack = 1. / render_hz - (time.perf_counter() - frame_start)
                plt.pause(max(.001, min(max(.001, slack), (playback.end - current) / speed)))
            if not loop:
                break
    finally:
        if not was_interactive:
            plt.ioff()
    return playback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", help="First run name, directory or ROM NPZ")
    parser.add_argument("run_b", help="Second run name, directory or ROM NPZ")
    parser.add_argument("--labels", nargs=2, help="Optional legend labels, in run order")
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1., help="Simulated seconds per wall second (default 1)")
    parser.add_argument("--render-hz", type=float, default=50.)
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--end", type=float)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--overlay", dest="overlay", action="store_true",
                        help="Draw both robots on one shared axis (default)")
    layout.add_argument("--side-by-side", dest="overlay", action="store_false",
                        help="Draw each robot on a separate axis")
    parser.set_defaults(overlay=True)
    parser.add_argument("--once", action="store_true", help="Play once rather than looping")
    parser.add_argument("--no-playback", action="store_true", help="Static plots only; usable with MPLBACKEND=Agg")
    parser.add_argument("--output-dir", type=Path, help="Save rom_states.png and rom_playback_preview.png")
    args = parser.parse_args(argv)
    try:
        if not np.isfinite([args.speed, args.render_hz]).all() or min(args.speed, args.render_hz) <= 0:
            raise ValueError("speed and render-hz must be finite and positive")
        runs = [load_rom_run(value, env_index=args.env_index) for value in (args.run_a, args.run_b)]
        if args.labels:
            runs = [replace(run, label=label) for run, label in zip(runs, args.labels)]
        elif runs[0].label == runs[1].label:
            runs = [replace(run, label=f"{run.label} ({Path(value).name})")
                    for run, value in zip(runs, (args.run_a, args.run_b))]
        start, end = comparison_window(runs, args.start, args.end)
        warn_if_unmatched(runs)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Comparing {runs[0].label} and {runs[1].label}: {start:g}–{end:g} simulated seconds")
    print(f"Playback: {args.speed:g}×; {(end - start) / args.speed:g} wall seconds per pass. No physics/controller is run.")
    static = plot_rom_comparison(runs, start=start, end=end)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        static.savefig(args.output_dir / "rom_states.png", dpi=160)
        preview = ROMComparisonPlayback(runs, start=start, end=end, overlay=args.overlay, speed=args.speed)
        preview.update(end)
        preview.fig.savefig(args.output_dir / "rom_playback_preview.png", dpi=160)
        plt.close(preview.fig)
        print(f"Saved comparison plots to {args.output_dir}")
    if not args.no_playback:
        if plt.get_backend().lower() in {"agg", "pdf", "svg", "ps", "cairo", "template"}:
            parser.error("Playback needs an interactive Matplotlib backend; use --no-playback for headless export")
        print("Close the playback window to stop. Solid triangles: measured; hollow gray: reference.")
        playback_rom_comparison(runs, start=start, end=end, speed=args.speed,
                                render_hz=args.render_hz, loop=not args.once, overlay=args.overlay)
    if args.no_playback and plt.get_backend().lower() == "agg":
        plt.close(static)
    else:
        plt.show()


if __name__ == "__main__":
    main()
