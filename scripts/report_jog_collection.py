"""Summarize collection coverage and achieved steady motion from full raw logs."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from utils.jog_commands import BEHAVIOR_NAMES, PROFILE_NAMES
from utils.math_utils import world_to_body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT/"data/data/jog_kcf_collection_v1")
    args = parser.parse_args()
    manifest = json.loads((args.directory/"full/manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Wait for the full collection and its quality checks to complete")
    samples = defaultdict(list)
    examples = {}
    totals = manifest["totals"]
    for shard in manifest["shards"]:
        with np.load(args.directory/"full"/shard["filename"], allow_pickle=False) as data:
            q, v = data["policy_q_log"], data["policy_v_log"]
            command, behavior = data["policy_cmd_log"], data["command_behavior_log"]
            valid = data["valid_transition_mask"]
            velocity = world_to_body(q[..., 3:7], v[..., :3])
            body = np.stack((velocity[..., 0], velocity[..., 1], v[..., 5]), axis=-1)
            profiles, variants = data["command_profile_labels"], data["command_profile_variants"]
            for profile in PROFILE_NAMES:
                candidates = np.flatnonzero(profiles == profile)
                if profile not in examples and len(candidates):
                    row = candidates[0]
                    examples[profile] = dict(command=command[row].copy(), measured=body[row].copy(),
                                             variant=str(variants[row]),
                                             trajectory_id=str(data["trajectory_ids"][row]))
            # Keep samples after at least one second in the same behavior.
            age = np.zeros(behavior.shape, dtype=np.int32)
            for index in range(1, behavior.shape[1]):
                age[:, index] = np.where(behavior[:, index] == behavior[:, index-1], age[:, index-1]+1, 0)
            settled = valid & (age >= 50)
            for index, name in enumerate(BEHAVIOR_NAMES):
                selected = settled & (behavior == index)
                pairs = np.concatenate((command[selected], body[selected]), axis=-1)
                # Deterministic thinning bounds plotting memory without altering raw data.
                if len(pairs):
                    samples[name].append(pairs[::max(1, len(pairs)//2000)])
    achieved = {}
    for name in BEHAVIOR_NAMES:
        values = np.concatenate(samples[name]) if samples[name] else np.empty((0, 6))
        if len(values):
            achieved[name] = dict(sampled_points=len(values),
                command_range=np.stack((values[:, :3].min(0), values[:, :3].max(0))).tolist(),
                measured_velocity_quantiles=np.quantile(values[:, 3:], [.01, .5, .99], axis=0).tolist(),
                velocity_rmse=np.sqrt(np.mean((values[:, 3:]-values[:, :3])**2, axis=0)).tolist())
    output = args.directory/"reports"
    output.mkdir(exist_ok=True)
    schema = {}
    with zipfile.ZipFile(args.directory/"full"/manifest["shards"][0]["filename"]) as archive:
        for name in archive.namelist():
            with archive.open(name) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError(f"Unsupported NPY header version: {version}")
            schema[name.removesuffix(".npy")] = dict(shape=list(shape), dtype=str(dtype), fortran_order=fortran)
    (output/"array_schema.json").write_text(json.dumps(schema, indent=2)+"\n")
    report = dict(status="complete", quality=manifest["quality"], totals=totals,
                  compressed_bytes=manifest["bytes"], achieved_motion=achieved,
                  measurement_note="Body velocities; samples after 1 s in a behavior; curves may contain speed changes",
                  kcf_branch=manifest["provenance"]["kcf_branch"],
                  relative_frame_sha256=manifest["provenance"]["source_sha256"]["utils/relative_frame.py"],
                  split_owner="KCF_MPC processing pipeline")
    (output/"collection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/robot-jog-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    matrix = np.asarray(totals["behavior_transitions"])
    heat = axes[0, 0].imshow(np.log10(matrix+1), cmap="viridis")
    axes[0, 0].set_xticks(range(7), BEHAVIOR_NAMES, rotation=40, ha="right")
    axes[0, 0].set_yticks(range(7), BEHAVIOR_NAMES)
    axes[0, 0].set_title("Valid directed transition counts")
    axes[0, 0].set_xlabel("To"); axes[0, 0].set_ylabel("From")
    for i in range(7):
        for j in range(7):
            if i != j:
                axes[0, 0].text(j, i, f"{matrix[i,j]:,}", ha="center", va="center", color="white", fontsize=8)
    profiles = totals["profiles"]
    axes[0, 1].barh(list(profiles), list(profiles.values()))
    axes[0, 1].set_title("Whole episodes by motion profile")
    axes[0, 1].set_xlabel("Episodes")
    axes[1, 0].bar(np.arange(43), totals["start_phase_counts"])
    axes[1, 0].set_title("Standing-to-motion events by gait phase")
    axes[1, 0].set_xlabel("Free-running clock index"); axes[1, 0].set_ylabel("Events")
    for name in ["forward", "backward"]:
        if samples[name]:
            values = np.concatenate(samples[name])
            axes[1, 1].scatter(values[:, 0], values[:, 3], s=3, alpha=.15, label=name)
    axes[1, 1].plot([-1, 1.5], [-1, 1.5], "k--", lw=1)
    axes[1, 1].set_title("Observed straight-motion tracking")
    axes[1, 1].set_xlabel("Command vx [m/s]"); axes[1, 1].set_ylabel("Measured body vx [m/s]")
    axes[1, 1].legend()
    figure.suptitle(f"Jogging collection: {totals['episodes']:,} episodes; {totals['valid_transitions']:,} valid transitions")
    figure.savefig(output/"coverage.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(len(PROFILE_NAMES), 2, figsize=(15, 19),
                                sharex=True, constrained_layout=True)
    for row, profile in enumerate(PROFILE_NAMES):
        example = examples[profile]
        command, measured = example["command"], example["measured"]
        time = np.arange(len(command))*.02
        standing = np.all(command == 0., axis=-1)
        for column, (component, label) in enumerate(((0, "Body vx [m/s]"), (2, "Yaw rate [rad/s]"))):
            axis = axes[row, column]
            axis.step(time, command[:, component], where="post", lw=1.2, color="black", label="Command")
            axis.plot(time, measured[:, component], lw=.9, alpha=.8, label="Measured")
            axis.fill_between(time, 0., 1., where=standing, transform=axis.get_xaxis_transform(),
                              color="grey", alpha=.10)
            axis.set_ylabel(label)
            axis.grid(alpha=.2)
            axis.set_title(f"{profile}: {example['variant']}", fontsize=10)
            if row == 0:
                axis.legend(loc="upper right")
    for axis in axes[-1]:
        axis.set_xlabel("Time [s]")
    figure.suptitle("One recorded episode per profile; shaded intervals have exact zero command")
    figure.savefig(output/"motion_examples.png", dpi=150)
    plt.close(figure)
    (output/"motion_examples.json").write_text(json.dumps(
        {name: {key: example[key] for key in ("variant", "trajectory_id")}
         for name, example in examples.items()}, indent=2)+"\n")
    print(json.dumps({"status":"complete", "episodes":totals["episodes"],
                      "valid_transitions":totals["valid_transitions"],
                      "quality":manifest["quality"], "reports":str(output)}, indent=2))


if __name__ == "__main__":
    main()
