"""Compare saved RL outputs, joint targets and spectra; never filter live commands.

previous_policy_action recorded at t is the output generated at t-control_dt.
Commands and measured joints are aligned to that output time before plotting.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import warnings
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from compare_rom_data import COLORS, load_rom_run, warn_if_unmatched
from policy.config import G1Config


@dataclass(frozen=True)
class PolicyRun:
    label: str
    times: np.ndarray
    commands: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    measured: np.ndarray
    dt: float
    gait_period: float
    configuration_verified: bool


def joint_names():
    """Use the actual qpos joint order, not alphabetical or actuator order."""
    body = ET.parse(ROOT / "models/g1_23dof.xml").getroot().find("worldbody")
    return [joint.attrib["name"].removesuffix("_joint") for joint in body.iter("joint")
            if joint.get("type", "hinge") not in {"free", "ball"}]


def verify_configuration(directory):
    """Only reconstruct historical target angles with the matching policy config."""
    path = directory / "pair_plan.json"
    hashes = json.loads(path.read_text()).get("source_hashes", {}) if path.is_file() else {}
    verified = True
    for name in ("policy/config.py", "models/g1_23dof.xml"):
        recorded = hashes.get(name)
        if recorded is None:
            verified = False
        elif hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != recorded:
            raise ValueError(f"{directory}: {name} differs from the recording; historical joint target reconstruction is unsafe")
    if not verified:
        warnings.warn(f"{directory.name}: no complete configuration hashes; joint targets use current G1Config (unverified historical scaling)",
                      stacklevel=2)
    return verified


def load_policy_run(directory, *, label=None):
    directory = Path(directory).expanduser()
    if not directory.is_dir():
        directory = ROOT / "data/data" / directory
    verified = verify_configuration(directory)
    cfg = G1Config()
    with np.load(directory / "g1_23dof_data.npz", allow_pickle=False) as data:
        q = np.asarray(data["q_log"], dtype=float)
        dt = float(data["control_dt"])
        log_dt = float(data["log_dt"] if "log_dt" in data else data["sim_dt"])
    with np.load(directory / "applied_policy_commands.npz", allow_pickle=False) as data:
        stamps = data["time"].astype(float)
        commands = data["command"].astype(float)
        previous = data["previous_policy_action"].astype(float)
    width = len(cfg.action_scale)
    if (stamps.ndim != 1 or len(stamps) < 3 or previous.shape != (len(stamps), width)
            or commands.shape != (len(stamps), 3) or q.ndim != 3
            or q.shape[0] != 1 or q.shape[2] != 7 + width or q.shape[1] < 2):
        raise ValueError("Expected single-environment full state and matching (T,23) policy actions / (T,3) commands")
    if (not np.isfinite([dt, log_dt]).all() or min(dt, log_dt) <= 0
            or not all(np.isfinite(x).all() for x in (stamps, previous, commands, q))
            or not np.allclose(np.diff(stamps), dt, rtol=1e-6, atol=1e-8)):
        raise ValueError("Policy records must be finite, uniformly sampled and match saved control_dt for spectral analysis")
    output_times = stamps - dt
    valid = output_times >= -1e-8  # Exclude the artificial pre-start history action.
    times = np.maximum(0., output_times[valid])
    actions = previous[valid]
    command_indices = np.searchsorted(stamps, times + 1e-8, side="right") - 1
    if np.any(command_indices < 0):
        raise ValueError("Command history does not cover the reconstructed output times")
    measured_times = np.arange(q.shape[1]) * log_dt
    if times[-1] > measured_times[-1] + 1e-8:
        raise ValueError("Measured joint log does not cover the policy output times")
    measured = np.column_stack([np.interp(times, measured_times, q[0, :, 7 + i]) for i in range(width)])
    targets = cfg.default_joint_pos + cfg.action_scale * actions
    plan_path = directory / "trial_plan.json"
    settings = json.loads(plan_path.read_text())["mpc"] if plan_path.is_file() else {}
    model = settings.get("prediction_model")
    default = {"kcf": "KCF", "single_integrator": "Single integrator"}.get(model, directory.name)
    return PolicyRun(label or default, times, commands[command_indices], actions, targets,
                     measured, dt, cfg.gait_period, verified)


def analysis_window(runs, start=10., end=None):
    lower = max(run.times[0] for run in runs)
    upper = min(run.times[-1] for run in runs)
    end = upper if end is None else end
    if not np.isfinite([start, end]).all() or not lower - 1e-8 <= start < end <= upper + 1e-8:
        raise ValueError(f"Require {lower:g} <= start < end <= {upper:g} (logged RL output times)")
    return float(start), float(end)


def density(values, dt):
    if len(values) < 8:
        raise ValueError("Need at least eight uniformly sampled actions for a spectrum")
    return welch(values, fs=1. / dt, axis=0, nperseg=min(512, len(values)),
                 detrend="constant", scaling="density")


def action_metrics(run, start, end, cutoff):
    selected = (run.times >= start - 1e-8) & (run.times <= end + 1e-8)
    actions, targets = run.actions[selected], run.targets[selected]
    frequency, power = density(actions, run.dt)
    if not np.isfinite(cutoff) or not 0 < cutoff < .5 / run.dt:
        raise ValueError("High-frequency cutoff must be positive and below the policy Nyquist frequency")
    total = power.sum(axis=0)
    fraction = np.divide(power[frequency > cutoff].sum(axis=0), total,
                         out=np.zeros_like(total), where=total > 1e-20)
    return dict(label=run.label, samples=int(selected.sum()), start_s=start, end_s=end,
                configuration_verified=run.configuration_verified,
                action_step_rms=np.sqrt(np.mean(np.diff(actions, axis=0)**2, axis=0)).tolist(),
                target_step_rms_rad=np.sqrt(np.mean(np.diff(targets, axis=0)**2, axis=0)).tolist(),
                high_frequency_power_fraction=fraction.tolist(), high_frequency_cutoff_hz=cutoff)


def plot_policy_comparison(runs, indices, *, start=10., end=None, zoom=6., cutoff=5.):
    """Short time traces plus full-window spectra and all-joint variation summary."""
    if len(runs) != 2 or not 1 <= len(indices) <= 6 or len(set(indices)) != len(indices):
        raise ValueError("Select two runs and one to six distinct joints")
    names = joint_names()
    if any(i < 0 or i >= len(names) for i in indices):
        raise ValueError("Joint indices must be in [0,22]")
    if not np.isfinite(zoom) or zoom <= 0:
        raise ValueError("Zoom duration must be finite and positive")
    start, end = analysis_window(runs, start, end)
    metrics = [action_metrics(run, start, end, cutoff) for run in runs]
    view_end = min(end, start + zoom)
    columns = max(3, len(indices))
    traces, axes = plt.subplots(4, columns, figsize=(4.4 * columns, 10), squeeze=False,
                               constrained_layout=True)
    spectra, frequency_axes = plt.subplots(2, columns, figsize=(4.4 * columns, 6), squeeze=False,
                                           constrained_layout=True)
    for run, color in zip(runs, COLORS):
        selected = (run.times >= start - 1e-8) & (run.times <= view_end + 1e-8)
        t = run.times[selected]
        for i, title in enumerate(("MPC vx [m/s]", "MPC vy [m/s]", "MPC yaw rate [rad/s]")):
            axes[0, i].step(t, run.commands[selected, i], where="post", color=color, label=run.label, linewidth=.9)
            axes[0, i].set_title(title)
        full = (run.times >= start - 1e-8) & (run.times <= end + 1e-8)
        f, command_power = density(run.commands[full], run.dt)
        _, action_power = density(run.actions[full], run.dt)
        for i in range(3):
            frequency_axes[0, i].semilogy(f, np.maximum(command_power[:, i], 1e-20), color=color,
                                         label=run.label)
        for column, index in enumerate(indices):
            axes[1, column].step(t, run.actions[selected, index], where="post", color=color,
                                 linewidth=.9, label=run.label)
            axes[1, column].set_title(f"{names[index]} — raw RL action")
            axes[2, column].step(t, run.targets[selected, index], where="post", color=color,
                                 linewidth=.9, label=f"{run.label} desired")
            axes[2, column].plot(t, run.measured[selected, index], "--", color=color,
                                 linewidth=.9, alpha=.8, label=f"{run.label} measured")
            axes[2, column].set_title("Desired / measured joint angle [rad]")
            delta_t = run.times[1:]
            visible_delta = (delta_t >= start - 1e-8) & (delta_t <= view_end + 1e-8)
            axes[3, column].plot(delta_t[visible_delta], np.diff(run.targets[:, index])[visible_delta],
                                 color=color, linewidth=.9)
            axes[3, column].set_title("Target change per policy tick [rad]")
            frequency_axes[1, column].semilogy(f, np.maximum(action_power[:, index], 1e-20),
                                              color=color, label=run.label)
            frequency_axes[1, column].set_title(f"{names[index]} — RL action PSD")
    for row in range(4):
        used = 3 if row == 0 else len(indices)
        for column in range(columns):
            ax = axes[row, column]
            if column >= used:
                ax.axis("off")
                continue
            ax.set(xlim=(start, view_end), xlabel="Simulation time [s]")
            ax.grid(True, alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[2, 0].legend(fontsize=7)
    for column, title in enumerate(("vx command PSD", "vy command PSD", "Yaw-rate command PSD")):
        frequency_axes[0, column].set_title(title)
    nyquist = min(.5 / run.dt for run in runs)
    for row in range(2):
        used = 3 if row == 0 else len(indices)
        for column in range(columns):
            ax = frequency_axes[row, column]
            if column >= used:
                ax.axis("off")
                continue
            ax.set(xlim=(0, nyquist), xlabel="Frequency [Hz]", ylabel="Power density [signal units²/Hz]")
            ax.axvspan(cutoff, nyquist, color="gray", alpha=.08)
            for period in sorted(set(run.gait_period for run in runs)):
                for harmonic in range(1, 6):
                    ax.axvline(harmonic / period, color="gray", linestyle=":", linewidth=.7)
            ax.grid(True, alpha=.25)
    frequency_axes[0, 0].legend(fontsize=8)
    traces.suptitle("RL output diagnostics — raw actions and physical joint targets; no smoothing applied")
    spectra.suptitle(f"Spectra over {start:g}–{end:g} s — dotted: gait rhythm/harmonics; shaded: >{cutoff:g} Hz (not automatically noise)")

    summary, bars = plt.subplots(1, 2, figsize=(12, 9), sharey=True, constrained_layout=True)
    y = np.arange(len(names))
    for i, (metric, color) in enumerate(zip(metrics, COLORS)):
        offset = (i - .5) * .36
        bars[0].barh(y + offset, metric["target_step_rms_rad"], height=.36, color=color, label=metric["label"])
        bars[1].barh(y + offset, 100 * np.asarray(metric["high_frequency_power_fraction"]), height=.36, color=color)
    bars[0].set_yticks(y, names, fontsize=8)
    bars[0].invert_yaxis()
    bars[0].set_xlabel("RMS desired-angle change per tick [rad]")
    bars[1].set_xlabel(f"RL action spectral power above {cutoff:g} Hz [%]")
    bars[0].legend(fontsize=8)
    for ax in bars:
        ax.grid(True, axis="x", alpha=.25)
    summary.suptitle(f"All 23 joints, {start:g}–{end:g} s — variation includes intended gait; not a noise verdict")
    return {"policy_time_traces": traces, "policy_spectra": spectra, "policy_joint_summary": summary}, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--start", type=float, default=10.)
    parser.add_argument("--end", type=float, help="Analysis end (default: last common recorded RL output)")
    parser.add_argument("--zoom", type=float, default=6., help="Time-trace duration; spectra use the full analysis interval")
    parser.add_argument("--joints", nargs="+", default=["left_hip_pitch", "left_knee", "left_ankle_pitch"],
                        help="One to six names or zero-based indices; all joints appear in the summary")
    parser.add_argument("--cutoff-hz", type=float, default=5., help="Illustrative high-frequency band, not a noise threshold")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-show", action="store_true", help="Save headlessly with MPLBACKEND=Agg")
    args = parser.parse_args(argv)
    try:
        names = joint_names()
        indices = [int(value) if value.isdecimal() else names.index(value.removesuffix("_joint")) for value in args.joints]
        runs = [load_policy_run(value) for value in (args.run_a, args.run_b)]
        # Preserve the existing comparison's provenance and task mismatch warnings.
        warn_if_unmatched([load_rom_run(value) for value in (args.run_a, args.run_b)])
        figures, metrics = plot_policy_comparison(runs, indices, start=args.start, end=args.end,
                                                  zoom=args.zoom, cutoff=args.cutoff_hz)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(args.output_dir / f"{name}.png", dpi=160)
        report = dict(joints=names, metrics=metrics,
                      interpretation="Higher-frequency power and action increments include gait harmonics and feedback; neither establishes noise or its cause.",
                      alignment="previous_policy_action[t] is plotted at t-control_dt; commands are zero-order held at that output time; initial history excluded")
        (args.output_dir / "policy_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved RL action diagnostics to {args.output_dir}")
    for metric in metrics:
        print(f"{metric['label']}: {metric['samples']} actions over {metric['start_s']:g}–{metric['end_s']:g} s; "
              f"all-joint target step RMS={np.sqrt(np.mean(np.asarray(metric['target_step_rms_rad'])**2)):.5f} rad")
    if args.no_show:
        for figure in figures.values():
            plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
