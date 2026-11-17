"""Plot measured smooth motion and assemble references to both raw collections."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from utils.jog_smooth_commands import PROFILE_NAMES
from utils.math_utils import world_to_body
from utils.relative_frame import RelativeFrame


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"data/data/jog_kcf_smooth_collection_v1")
    parser.add_argument("--phase", choices=["pilot", "full"], default="full")
    args = parser.parse_args()
    source = args.directory/args.phase
    manifest_path = source/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or not manifest["quality"]["passed"]:
        raise RuntimeError("Collection must finish and pass its quality checks first")
    output = args.directory/("pilot_reports" if args.phase == "pilot" else "reports")
    output.mkdir(exist_ok=True)
    examples, squared, counts, loop_errors = {}, defaultdict(lambda: np.zeros(3)), defaultdict(int), defaultdict(list)
    identities, trajectory_ids = [], []
    for shard in manifest["shards"]:
        path = source/shard["filename"]
        if digest(path) != shard["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            q, v = data["q_log"], data["v_log"]
            command, valid = data["policy_cmd_log"], data["valid_transition_mask"]
            labels, variants = data["command_profile_labels"], data["command_profile_variants"]
            identities.extend(data["episode_ids"].tolist())
            trajectory_ids.extend(data["trajectory_ids"].tolist())
            if not np.isfinite(data["terminal_v"]).all():
                raise ValueError("Nonfinite terminal velocity")
            velocity = world_to_body(q[:, :-1, 3:7], v[:, :-1, :3])
            measured = np.stack((velocity[..., 0], velocity[..., 1], v[:, :-1, 5]), axis=-1)
            for profile in PROFILE_NAMES:
                rows = np.flatnonzero(labels == profile)
                selected = valid & (labels[:, None] == profile)
                error = measured[selected]-command[selected]
                squared[profile] += np.sum(error**2, axis=0)
                counts[profile] += len(error)
                errors = data["measured_loop_closure_error_m"][rows]
                loop_errors[profile].extend(errors[errors >= 0.].tolist())
                # Keep both traversal directions for the figure-eight comparison.
                for direction in (["forward", "backward"] if profile == "figure_eight" else [None]):
                    key = profile if direction is None else f"{profile}_{direction}"
                    candidates = rows if direction is None else rows[np.char.startswith(variants[rows], direction)]
                    if key in examples or not len(candidates):
                        continue
                    row = candidates[0]
                    frame = RelativeFrame.from_pose([*q[row, 0, :2], 0.], q[row, 0, 3:7], yaw_only=True)
                    examples[key] = dict(profile=profile, variant=str(variants[row]),
                        trajectory_id=str(data["trajectory_ids"][row]), command=command[row].copy(),
                        measured=measured[row].copy(), actual=frame.position_to_local(q[row, :, :3])[:, :2],
                        reference=data["command_integrated_reference_log"][row].copy(),
                        core_start=int(data["path_core_start_query"][row]),
                        period=float(data["path_period_s"][row]))
        print(f"Verified and summarized {shard['filename']}", flush=True)
    if len(identities) != len(set(identities)) or len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Duplicate trajectory IDs")
    offset = 3_000_000 if args.phase == "pilot" else 2_000_000
    np.testing.assert_array_equal(identities, np.arange(offset, offset+manifest["totals"]["episodes"]))
    np.testing.assert_array_equal(trajectory_ids,
        [f"{manifest['config']['collection_id']}:{identity}" for identity in identities])
    assert sum(counts.values()) == manifest["totals"]["valid_transitions"]
    for name, expected in manifest["provenance"]["source_sha256"].items():
        if digest(ROOT/name) != expected:
            raise ValueError(f"Source fingerprint changed: {name}")
    assert digest(ROOT/"models/g1_29dof/g1_29dof.xml") == manifest["provenance"]["model_sha256"]
    assert digest(ROOT/"policy/jog/g1_29dof_jog_lstm_v0_cont.onnx") == manifest["provenance"]["policy_sha256"]
    report = dict(phase=args.phase, quality=manifest["quality"], totals=manifest["totals"],
        compressed_bytes=manifest["bytes"], all_file_checksums_verified=True, unique_trajectories=len(trajectory_ids),
        velocity_tracking_rmse={name:np.sqrt(squared[name]/counts[name]).tolist() for name in PROFILE_NAMES},
        measured_loop_closure_error_quantiles_m={name:np.quantile(values, [.1, .5, .9, .99]).tolist()
            for name, values in loop_errors.items() if values},
        measurement_note="Tracking/loop errors measure the simulated policy, not KCF prediction accuracy; all valid time samples included",
        reference_note="Dashed paths integrate feedforward body commands in the initial ground/yaw frame; no pose is imposed on physics",
        traversal_note="Periodic variant suffixes identify parameterization direction; randomized arc offsets determine initial turn sign",
        kcf_branch=manifest["provenance"]["kcf_branch"], split_owner="KCF_MPC processing pipeline")
    (output/"collection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/robot-jog-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = ["figure_eight_forward", "figure_eight_backward", *PROFILE_NAMES[1:]]
    figure, axes = plt.subplots(4, 2, figsize=(14, 18), constrained_layout=True)
    for key, axis in zip(keys, axes.flat):
        example = examples[key]
        reference, actual = example["reference"], example["actual"]
        axis.plot(reference[:, 0], reference[:, 1], "k--", lw=1.3, label="Command-integrated reference")
        axis.plot(actual[:, 0], actual[:, 1], lw=1., alpha=.85, label="Simulated robot")
        axis.plot([0], [0], "go", ms=5, label="Start")
        description = example["variant"]
        if example["period"] > 0.:
            direction = "backward" if description.startswith("backward") else "forward"
            turning = example["command"][:, 2]
            first_turn = turning[np.flatnonzero(np.abs(turning) > 1e-5)[0]]
            description = f"{direction}; initial turn {'left' if first_turn > 0 else 'right'}"
        axis.set_title(f"{example['profile']}: {description}")
        axis.set_xlabel("Local x [m]"); axis.set_ylabel("Local y [m]")
        axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"Smooth jogging {args.phase}: {manifest['totals']['episodes']:,} trajectories; actual paths and nominal references")
    figure.savefig(output/"smooth_paths.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(len(PROFILE_NAMES), 2, figsize=(15, 18), sharex=True, constrained_layout=True)
    for row, name in enumerate(PROFILE_NAMES):
        example = examples["figure_eight_forward" if name == "figure_eight" else name]
        command, measured = example["command"], example["measured"]
        time = np.arange(len(command))*.02
        standing = np.all(command == 0., axis=-1)
        for column, (component, label) in enumerate(((0, "Body vx [m/s]"), (2, "Yaw rate [rad/s]"))):
            axis = axes[row, column]
            axis.plot(time, command[:, component], color="black", lw=1.3, label="Command")
            axis.plot(time, measured[:, component], lw=.7, alpha=.65, label="Measured")
            axis.fill_between(time, 0., 1., where=standing, transform=axis.get_xaxis_transform(), color="grey", alpha=.10)
            axis.set_ylabel(label); axis.set_title(name); axis.grid(alpha=.2)
            if row == 0:
                axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time [s]")
    figure.suptitle("Smooth speed and turn-rate inputs; zero-command mode changes remain explicit")
    figure.savefig(output/"smooth_commands.png", dpi=140)
    plt.close(figure)
    (output/"example_trajectories.json").write_text(json.dumps(
        {name:dict(trajectory_id=value["trajectory_id"], variant=value["variant"]) for name, value in examples.items()}, indent=2)+"\n")

    if args.phase == "full":
        old_path = ROOT/"data/data/jog_kcf_collection_v1/full/manifest.json"
        old = json.loads(old_path.read_text())
        assert old["status"] == "complete" and old["quality"]["passed"]
        for key in ("model_sha256", "policy_sha256"):
            assert old["provenance"][key] == manifest["provenance"][key]
        assert old["provenance"]["source_sha256"]["utils/relative_frame.py"] == manifest["provenance"]["source_sha256"]["utils/relative_frame.py"]
        sources = []
        for kind, item_path, item in (("holds_and_transitions", old_path, old), ("smooth_paths", manifest_path, manifest)):
            sources.append(dict(kind=kind, raw_directory=str(item_path.parent.resolve()),
                manifest=str(item_path.resolve()), manifest_sha256=digest(item_path),
                collection_id=item["config"]["collection_id"], episodes=item["totals"]["episodes"],
                valid_transitions=item["totals"]["valid_transitions"], duration_s=item["config"]["duration_s"]))
        corpus = dict(collection_id="jog_kcf_combined_v1", sources=sources,
            episodes=sum(item["episodes"] for item in sources),
            valid_transitions=sum(item["valid_transitions"] for item in sources),
            smooth_supplement_transition_fraction=manifest["totals"]["valid_transitions"]/sum(item["valid_transitions"] for item in sources),
            policy_log_schema="g1_jog_policy_v1", control_dt=.02,
            kcf_branch=manifest["provenance"]["kcf_branch"],
            relative_frame_sha256=manifest["provenance"]["source_sha256"]["utils/relative_frame.py"],
            split_owner="KCF_MPC processing pipeline; split whole trajectories before making windows",
            note="Raw source directories only; pilots excluded; use collection ID and episode ID together")
        (args.directory/"combined_corpus.json").write_text(json.dumps(corpus, indent=2)+"\n")
    print(json.dumps(dict(phase=args.phase, episodes=report["unique_trajectories"],
                         quality=report["quality"], reports=str(output)), indent=2), flush=True)


if __name__ == "__main__":
    main()
