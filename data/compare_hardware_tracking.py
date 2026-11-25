"""Compare fixed-target MPC hardware or simulation traces at their original time scales.

Example (from the repository root, in the deploy environment)::

    python data/compare_hardware_tracking.py TRACE_KCF.npz TRACE_SI.npz \
        --hardware-logs LOG_KCF.h5 LOG_SI.h5 --output-dir OUTPUT

Offsets are seconds after each trace's first recorded state. Both selected
windows have the same duration. Plots retain measured world coordinates and
native samples; all distance metrics use the final fixed target, including
during the logged startup reference ramp.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


COLORS = ("#1768ac", "#d55e00")


@dataclass
class Trace:
    path: Path
    label: str
    settings: dict
    times: np.ndarray
    measured: np.ndarray
    reference: np.ndarray
    generation: np.ndarray

    @property
    def goal(self):
        return np.asarray(self.settings["reference"][:3], dtype=float)

    @property
    def error(self):
        return np.linalg.norm(self.measured[:, :2] - self.goal[:2], axis=1)

    @property
    def yaw_error_deg(self):
        yaw = np.arctan2(self.measured[:, 2], self.measured[:, 3])
        difference = yaw - self.goal[2]
        return np.degrees(np.arctan2(np.sin(difference), np.cos(difference)))


def load_trace(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        settings = json.loads(str(data["metadata"]))["settings"]
        trace = Trace(
            path=path,
            label={"kcf": "KCF", "unicycle": "Unicycle"}.get(
                settings.get("prediction_model"), path.stem
            ),
            settings=settings,
            times=data["state_time"].astype(float),
            measured=data["measured_observation"].astype(float),
            reference=data["reference"][:, 0].astype(float),
            generation=data["generation"].copy(),
        )
    if settings.get("reference_mode") != "fixed":
        raise ValueError(f"{path}: this comparison requires a fixed target")
    n = len(trace.times)
    if (n < 2 or trace.measured.ndim != 2 or trace.measured.shape[0] != n
            or trace.measured.shape[1] < 7 or trace.reference.shape != (n, 10)
            or trace.generation.shape != (n,)):
        raise ValueError(f"{path}: inconsistent native-85 trace arrays")
    if (not np.isfinite(trace.times).all() or np.any(np.diff(trace.times) <= 0)
            or not np.isfinite(trace.measured[:, :7]).all()
            or not np.isfinite(trace.reference).all() or not np.isfinite(trace.goal).all()):
        raise ValueError(f"{path}: require finite states and strictly increasing times")
    return trace


def select_window(trace, offset, duration):
    start = trace.times[0] + offset
    end = start + duration
    indices = []
    for boundary in (start, end):
        index = int(np.argmin(np.abs(trace.times - boundary)))
        if abs(trace.times[index] - boundary) > 1e-7:
            raise ValueError(
                f"{trace.label}: boundary {boundary:g}s is not a recorded sample; "
                "choose offsets/duration on the recorded time grid"
            )
        indices.append(index)
    first, last = indices
    selected = slice(first, last + 1)
    if (last <= first or np.max(np.diff(trace.times[selected])) > 0.1 + 1e-7
            or np.unique(trace.generation[selected]).size != 1):
        raise ValueError(f"{trace.label}: selected window has a gap or tracking reset")
    return selected


def band_hold_start(times, errors, tolerance, hold_seconds=1.0):
    inside = errors <= tolerance
    edges = np.diff(np.r_[False, inside, False].astype(int))
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if times[stop - 1] - times[first] >= hold_seconds - 1e-7:
            return float(times[first])
    return None


def time_mean(times, values):
    return float(np.trapezoid(values, times) / (times[-1] - times[0]))


def window_metrics(trace, selected, hold_from):
    t = trace.times[selected] - trace.times[selected][0]
    error = trace.error[selected]
    entry = np.flatnonzero(error <= 0.05)
    hold = t >= hold_from - 1e-7
    return {
        "model": trace.label,
        "initial_goal_distance_m": float(error[0]),
        "initial_x_m": float(trace.measured[selected][0, 0]),
        "initial_y_m": float(trace.measured[selected][0, 1]),
        "initial_yaw_error_deg": float(trace.yaw_error_deg[selected][0]),
        "first_entry_5cm_s": float(t[entry[0]]) if entry.size else None,
        "first_1s_hold_within_5cm_start_s": band_hold_start(t, error, 0.05),
        "mean_goal_error_hold_m": time_mean(t[hold], error[hold]),
        "rmse_goal_error_hold_m": float(np.sqrt(time_mean(t[hold], error[hold] ** 2))),
        "max_goal_error_hold_m": float(np.max(error[hold])),
        "rmse_goal_error_whole_window_m": float(np.sqrt(time_mean(t, error ** 2))),
        "final_goal_error_m": float(error[-1]),
    }


def source_window(trace, selected, hardware_log=None):
    first, stop = selected.start, selected.stop
    start, end = trace.times[first], trace.times[stop - 1]
    info = {
        "label": trace.label,
        "trace": str(trace.path),
        "trace_row_start_inclusive": first,
        "trace_row_end_inclusive": stop - 1,
        "samples": stop - first,
        "hardware_time_start_s": float(start),
        "hardware_time_end_s": float(end),
        "trace_relative_start_s": float(start - trace.times[0]),
        "trace_relative_end_s": float(end - trace.times[0]),
        "aligned_time_formula": f"t_aligned = state_time - {start:.8f}",
        "input_lower_bounds": trace.settings.get("input_lower_bounds"),
        "input_upper_bounds": trace.settings.get("input_upper_bounds"),
    }
    if hardware_log is not None:
        import h5py

        hardware_log = Path(hardware_log).resolve()
        with h5py.File(hardware_log, "r") as data:
            hardware_times = np.asarray(data["time"]).reshape(-1).astype(float)
        if (not np.isfinite(hardware_times).all() or np.any(np.diff(hardware_times) < 0)
                or start < hardware_times[0] or end > hardware_times[-1]):
            raise ValueError(f"{hardware_log}: selected window is outside the hardware log")
        info.update(
            hardware_log=str(hardware_log),
            hardware_log_first_time_s=float(hardware_times[0]),
            seconds_after_first_hardware_log_sample=[
                float(start - hardware_times[0]), float(end - hardware_times[0])
            ],
        )
    return info


def plot_comparison(traces, windows, metrics, duration, hold_from, output):
    fig = plt.figure(figsize=(13.5, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.15))
    xy = fig.add_subplot(grid[:2, 0])
    xax = fig.add_subplot(grid[0, 1])
    yax = fig.add_subplot(grid[1, 1], sharex=xax)
    error_ax = fig.add_subplot(grid[2, 0])
    yaw_ax = fig.add_subplot(grid[2, 1], sharex=error_ax)
    goal = traces[0].goal
    starts = [trace.measured[selected.start, :2] for trace, selected in zip(traces, windows)]
    shared_start = np.allclose(starts[0], starts[1], rtol=0, atol=1e-6)
    xy.add_patch(Circle(goal[:2], 0.05, color="#999999", alpha=0.15))
    xy.plot(*goal[:2], "*", color="#222222", ms=15, label="Fixed target")
    for trace, selected, metric, color in zip(traces, windows, metrics, COLORS):
        times = trace.times[selected] - trace.times[selected][0]
        measured = trace.measured[selected]
        reference = trace.reference[selected]
        error = trace.error[selected]
        xy.plot(reference[:, 0], reference[:, 1], color="black", ls=":", lw=1.5,
                label=f"{trace.label} reference")
        xy.plot(measured[:, 0], measured[:, 1], color=color, lw=1.5,
                label=f"{trace.label} measured")
        xy.plot(*measured[0, :2], "o", color=color, ms=7)
        xy.plot(*measured[-1, :2], "s", color=color, ms=6)
        if not shared_start:
            xy.annotate(f"{trace.label} start", measured[0, :2], xytext=(0, 13),
                        textcoords="offset points", fontsize=9, color=color)
        for ax, column in ((xax, 0), (yax, 1)):
            ax.plot(times, measured[:, column], color=color, lw=1.5)
            ax.plot(times, reference[:, column], color="black", ls=":", lw=1.5)
        error_ax.plot(times, error, color=color, lw=1.5)
        yaw_ax.plot(times, trace.yaw_error_deg[selected], color=color, lw=1.5)
        entry = metric["first_entry_5cm_s"]
        if entry is not None:
            index = int(np.argmin(abs(times - entry)))
            error_ax.plot(entry, error[index], "o", ms=5, color=color)
    if shared_start:
        xy.annotate("Shared start", starts[0], xytext=(0, 13),
                    textcoords="offset points", fontsize=9, color="black")
    xy.set(xlabel="World x [m]", ylabel="World y [m]", title="Tracked reference")
    xy.set_aspect("equal", adjustable="datalim")
    xy.legend(loc="lower left", bbox_to_anchor=(0.01, 0.08), fontsize=9)
    xy.text(0.02, 0.02, "Circles: start    Squares: end", transform=xy.transAxes, fontsize=9)
    xax.set(ylabel="World x [m]", title="Position versus aligned time")
    yax.set(ylabel="World y [m]", xlabel="Aligned time [s]")
    for ax, coordinate in ((xax, 0), (yax, 1)):
        ax.axhline(goal[coordinate], color="#333333", lw=1.5, ls="--")
        ax.set_xlim(0, duration)
    xax.text(0.98, 0.03, "Dotted: logged first-horizon reference\nDashed: final fixed target",
             ha="right", va="bottom", transform=xax.transAxes, fontsize=8)
    error_ax.axhline(0.05, color="#666666", ls="--", lw=1.5)
    error_ax.set(xlabel="Aligned time [s]", ylabel="Distance to final target [m]",
                 title="Approach and target hold",
                 xlim=(0, duration), ylim=(0, None))
    yaw_ax.axhline(0, color="#333333", lw=1.5, ls="--")
    yaw_ax.set(xlabel="Aligned time [s]", ylabel="Wrapped heading error [deg]",
               title="Heading relative to target yaw", xlim=(0, duration))
    for ax in (xy, xax, yax, error_ax, yaw_ax):
        ax.grid(True, alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed target tracking", fontsize=14)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"tracking_comparison.{extension}", dpi=200)
    plt.close(fig)


def plot_overview(traces, windows, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.5), layout="constrained")
    for column, (trace, selected, color) in enumerate(zip(traces, windows, COLORS)):
        times = trace.times - trace.times[0]
        axes[0, column].plot(times, trace.error, color=color, lw=1.5)
        axes[1, column].plot(times, trace.yaw_error_deg, color=color, lw=1.5)
        for ax in axes[:, column]:
            ax.axvspan(times[selected.start], times[selected.stop - 1], color=color,
                       alpha=0.15, label="Selected comparison window")
            ax.set_xlim(0, times[-1])
            ax.grid(True, alpha=0.22)
        axes[0, column].set_title(f"{trace.label}: {trace.path.name}", fontsize=11)
        axes[0, column].set_ylabel("Distance to final target [m]")
        axes[0, column].legend(fontsize=8)
        axes[1, column].set(ylabel="Heading error [deg]",
                            xlabel="Seconds from first recorded tracking sample")
    fig.suptitle("Full recorded traces, with the selected windows shaded", fontsize=14)
    fig.savefig(output / "selection_overview.png", dpi=180)
    plt.close(fig)


def write_outputs(traces, windows, metrics, selection, output):
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    with (output / "aligned_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "trace_row", "aligned_time_s", "hardware_time_s", "x_m", "y_m",
                         "yaw_error_deg", "distance_to_final_target_m", "logged_ref_x_m",
                         "logged_ref_y_m"))
        for trace, selected in zip(traces, windows):
            start = trace.times[selected.start]
            for index in range(selected.start, selected.stop):
                writer.writerow((trace.label, index, trace.times[index] - start, trace.times[index],
                                 *trace.measured[index, :2], trace.yaw_error_deg[index],
                                 trace.error[index], *trace.reference[index, :2]))
    notes = [
        "VIDEO ALIGNMENT AND SELECTED SECTIONS",
        f"Comparison duration: {selection['duration_s']:g} seconds at original speed.",
        "Alignment: set each selected window's first tracking state to t = 0.",
        "Selection: initial approach and subsequent target hold.",
        "The full traces are shown in selection_overview.png for context.",
        "",
    ]
    for info, metric in zip(selection["windows"], metrics):
        entry = metric["first_entry_5cm_s"]
        hold = metric["first_1s_hold_within_5cm_start_s"]
        notes.extend([
            info["label"],
            f"  Trace: {info['trace']}",
            f"  Raw state_time: {info['hardware_time_start_s']:.2f} to "
            f"{info['hardware_time_end_s']:.2f} seconds.",
            f"  Seconds from trace start: {info['trace_relative_start_s']:.2f} to "
            f"{info['trace_relative_end_s']:.2f}.",
            f"  Source rows (zero-based, inclusive): {info['trace_row_start_inclusive']} to "
            f"{info['trace_row_end_inclusive']}.",
            f"  Initial measured position: ({metric['initial_x_m']:.4f}, "
            f"{metric['initial_y_m']:.4f}) m.",
            f"  First entry within 5 cm of target: t = {entry:.2f} s."
            if entry is not None else "  No entry within 5 cm in this window.",
            f"  First continuous 1-second hold inside 5 cm starts at t = {hold:.2f} s."
            if hold is not None else "  No continuous 1-second hold inside 5 cm in this window.",
        ])
        if "hardware_log" in info:
            offsets = info["seconds_after_first_hardware_log_sample"]
            notes.extend([
                f"  Hardware log: {info['hardware_log']}",
                f"  Seconds after first hardware-log time sample: {offsets[0]:.2f} to {offsets[1]:.2f}.",
            ])
        notes.append("")
    notes.extend([
        "VIDEO CUTS",
        "The offsets above use log clocks. They are not confirmed timestamps in the video files.",
        "Match the initial tracking activation/approach in each video to t = 0, then retain 10 seconds"
        if selection["duration_s"] == 10 else
        f"Match each selected tracking state to t = 0, then retain {selection['duration_s']:g} seconds",
        "at original speed. Keep the same playback rate; do not align at arrival or stretch time.",
        "With a known video/log sync anchor: video_time = anchor_video_time + "
        "(hardware_time - anchor_hardware_time).",
        "Do not infer exact wall-clock cuts by adding offsets to the hardware log filename:",
        "the filename is a logger-start label, not a calibrated video sync marker.",
        "",
        "METHOD",
        "Use measured_observation (measured feedback), not latency-projected observation or predicted plans.",
        "Preserve world coordinates, raw samples and the original clock. No smoothing or time warping.",
        "Distance metrics refer to the final fixed target, even while the startup reference ramps.",
        "Dotted position curves show the recorded first-horizon reference at each solve's state time.",
        "The first continuous 1-second hold is not a claim of permanent settling.",
        f"Target-hold metrics use the same aligned {selection['hold_metrics_from_s']:g} to "
        f"{selection['duration_s']:g} second interval in both recordings.",
        "Mean/RMSE metrics integrate over recorded time (trapezoidal weighting).",
        "Initial positions differ; the figure does not translate either trajectory to hide this.",
    ])
    (output / "video_edit_notes.txt").write_text("\n".join(notes) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs=2, type=Path, metavar="TRACE")
    parser.add_argument("--hardware-logs", nargs=2, type=Path, metavar="H5")
    parser.add_argument("--offsets", nargs=2, type=float, default=(0.0, 0.0), metavar="SECONDS")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hold-metrics-from", type=float, default=5.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (not np.isfinite([args.duration, args.hold_metrics_from, *args.offsets]).all()
            or not 0 <= args.hold_metrics_from < args.duration or min(args.offsets) < 0):
        parser.error("require finite duration > hold-metrics-from >= 0 and offsets >= 0")
    traces = [load_trace(path) for path in args.traces]
    if not np.allclose(traces[0].goal, traces[1].goal, rtol=0, atol=1e-9):
        parser.error("the traces have different fixed targets")
    windows = [select_window(trace, offset, args.duration)
               for trace, offset in zip(traces, args.offsets)]
    for trace, selected in zip(traces, windows):
        times = trace.times[selected] - trace.times[selected.start]
        if np.count_nonzero(times >= args.hold_metrics_from - 1e-7) < 2:
            parser.error("the target-hold interval needs at least two recorded samples")
    metrics = [window_metrics(trace, selected, args.hold_metrics_from)
               for trace, selected in zip(traces, windows)]
    hardware_logs = args.hardware_logs or (None, None)
    selection = {
        "duration_s": args.duration,
        "hold_metrics_from_s": args.hold_metrics_from,
        "target_world_x_y_yaw": traces[0].goal.tolist(),
        "alignment": "first selected recorded tracking state; offset only, no time scaling",
        "selection_reason": "initial approach and subsequent target hold",
        "windows": [source_window(trace, selected, log)
                    for trace, selected, log in zip(traces, windows, hardware_logs)],
        "metrics": metrics,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_comparison(traces, windows, metrics, args.duration, args.hold_metrics_from, output)
    plot_overview(traces, windows, output)
    write_outputs(traces, windows, metrics, selection, output)
    print(json.dumps(selection, indent=2))
    print(f"Saved comparison and video edit notes to {output}")


if __name__ == "__main__":
    main()
