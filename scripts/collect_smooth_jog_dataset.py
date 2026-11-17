"""Collect a separate smooth-input supplement with the original policy logger."""

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["ROBOT_SIM_ROOT_DIR"] = str(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.collect_jog_dataset import aggregate, atomic_json, audit, compressed_archive, digest
from sim.parallel.parallel_sim_jog import JogSimConfig, ParallelJogSim, MODE_NAMES
from utils.jog_commands import BEHAVIOR_NAMES
from utils.jog_smooth_commands import PROFILE_NAMES, make_smooth_jog_program


def smooth_audit(data, policy=None):
    stats = audit(data, policy)
    command, valid = data["policy_cmd_log"], data["valid_transition_mask"]
    delta = np.abs(np.diff(command, axis=1))
    same_drive = command[:, 1:, 0]*command[:, :-1, 0] > 0.
    same_pivot = ((command[:, 1:, 0] == 0.) & (command[:, :-1, 0] == 0.)
                  & (command[:, 1:, 2]*command[:, :-1, 2] > 0.))
    pairs = valid[:, 1:] & valid[:, :-1]
    smooth = pairs & (same_drive | same_pivot)
    changed = delta > 1e-7
    stats["varying_same_mode_pairs"] = int((smooth & changed.any(-1)).sum())
    stats["jointly_varying_drive_pairs"] = int((pairs & same_drive & changed[..., 0] & changed[..., 2]).sum())
    stats["same_mode_max_delta"] = delta[smooth].max(0).tolist() if smooth.any() else [0.]*3
    labels, variants = data["command_profile_labels"], data["command_profile_variants"]
    stats["backward_episodes"] = int(np.char.startswith(variants, "backward").sum())
    loops = Counter()
    nominal_error, actual_error = [], []
    data["nominal_loop_closure_error_m"] = np.full(len(command), -1., dtype=np.float64)
    data["measured_loop_closure_error_m"] = np.full(len(command), -1., dtype=np.float64)
    reference = data["command_integrated_reference_log"]
    if reference.shape != (command.shape[0], command.shape[1]+1, 3) or not np.isfinite(reference).all():
        raise ValueError("Invalid command-integrated reference")
    for row, period in enumerate(data["path_period_s"]):
        if period <= 0.:
            continue
        start = int(data["path_core_start_query"][row])
        stop = start+period/.02
        lower = int(np.floor(stop))
        weight = stop-lower
        if lower+1 > data["path_core_end_query"][row]:
            raise ValueError("Periodic path is truncated before one full cycle")
        target = (1.-weight)*reference[row, lower, :2]+weight*reference[row, lower+1, :2]
        error = float(np.linalg.norm(target-reference[row, start, :2]))
        data["nominal_loop_closure_error_m"][row] = error
        nominal_error.append(error)
        q = data["q_log"]
        target = (1.-weight)*q[row, lower, :2]+weight*q[row, lower+1, :2]
        error = float(np.linalg.norm(target-q[row, start, :2]))
        data["measured_loop_closure_error_m"][row] = error
        actual_error.append(error)
        if valid[row, start:lower+1].all():
            loops[str(labels[row])] += 1
    stats["complete_valid_reference_loops"] = dict(loops)
    stats["maximum_nominal_loop_error_m"] = max(nominal_error, default=0.)
    stats["maximum_measured_loop_error_m"] = max(actual_error, default=0.)
    return stats


def totals(entries):
    result = aggregate(entries)
    for key in ("varying_same_mode_pairs", "jointly_varying_drive_pairs", "backward_episodes"):
        result[key] = sum(item["audit"][key] for item in entries)
    for key in ("maximum_nominal_loop_error_m", "maximum_measured_loop_error_m"):
        result[key] = max((item["audit"][key] for item in entries), default=0.)
    result["same_mode_max_delta"] = np.max([item["audit"]["same_mode_max_delta"] for item in entries], axis=0).tolist()
    loops = Counter()
    for item in entries:
        loops.update(item["audit"]["complete_valid_reference_loops"])
    result["complete_valid_reference_loops"] = dict(loops)
    return result


def quality(stats, config, phase):
    failures = []
    maximum = config["pilot_maximum_failure_fraction"] if phase == "pilot" else config["maximum_failure_fraction"]
    fraction = stats["failed_episodes"]/stats["episodes"]
    if fraction > maximum:
        failures.append("Too many episodes cross the existing height/tilt envelope")
    if any(stats["profiles"].get(name, 0) == 0 for name in PROFILE_NAMES):
        failures.append("Missing smooth motion profiles")
    if min(stats["behaviors"]) == 0:
        failures.append("Missing forward/backward/curve/pivot behaviors")
    if stats["varying_same_mode_pairs"] < .5*stats["queries"]:
        failures.append("Insufficient continuously varying command coverage")
    if stats["jointly_varying_drive_pairs"] < .3*stats["queries"]:
        failures.append("Insufficient simultaneous speed and yaw variation")
    if stats["maximum_nominal_loop_error_m"] > .02:
        failures.append("Command-integrated periodic paths do not close within 2 cm")
    for name in ("figure_eight", "ovals"):
        if stats["complete_valid_reference_loops"].get(name, 0) < (1.-maximum)*stats["profiles"].get(name, 0):
            failures.append(f"Insufficient complete valid {name} cycles")
    if phase == "full" and (min(stats["initial_phase_counts"]) == 0 or min(stats["start_phase_counts"]) == 0):
        failures.append("Missing initial or restart gait phases")
    return dict(passed=not failures, failures=failures, episode_failure_fraction=fraction,
                note="Nominal paths are integrated feedforward commands; actual tracking is reported separately")


def run_phase(config, root, phase, provenance):
    directory = root/phase
    directory.mkdir(exist_ok=True)
    batch = config["pilot_batch_size"] if phase == "pilot" else config["batch_size"]
    batches = 1 if phase == "pilot" else config["batches"]
    offset = 3_000_000 if phase == "pilot" else 2_000_000
    fingerprint = hashlib.sha256(json.dumps(dict(config=config, phase=phase, provenance=provenance), sort_keys=True).encode()).hexdigest()
    path = directory/"manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else dict(
        phase=phase, fingerprint=fingerprint, config=config, provenance=provenance, shards=[])
    if manifest["fingerprint"] != fingerprint:
        raise ValueError("Settings/source changed; use a new output directory")
    for entry in manifest["shards"]:
        if digest(directory/entry["filename"]) != entry["sha256"]:
            raise ValueError("A saved shard checksum changed")
    if manifest.get("status") == "complete":
        return manifest
    sim = ParallelJogSim(JogSimConfig(batch_size=batch, seed=config["seed"],
        xml_path=str(ROOT/"models/g1_29dof/g1_29dof.xml"),
        policy_path=str(ROOT/"policy/jog/g1_29dof_jog_lstm_v0_cont.onnx"), log_hz=50,
        init_joint_noise=config["init_joint_noise"], init_yaw_range=config["init_yaw_range"],
        init_xy_range=config["init_xy_range"], policy_threads=config["policy_threads"]))
    queries = int(round(config["duration_s"]/.02))
    manifest.update(status="running", target_episodes=batch*batches, target_shards=batches)
    atomic_json(path, manifest)
    for index in range(len(manifest["shards"]), batches):
        started = time.monotonic()
        if shutil.disk_usage(root).free < config["minimum_free_gb"]*1e9 + batch*queries*3500:
            raise OSError("Insufficient free space for the next smooth batch")
        filename = f"g1_29dof_jog_smooth_data_{index:05d}.npz"
        archive_path = directory/filename
        rollout_seed = config["seed"]+index+(100_000 if phase == "pilot" else 0)
        if archive_path.exists():
            with np.load(archive_path, allow_pickle=False) as data:
                payload = {name:data[name] for name in data.files}
            if payload["collection_fingerprint"].item() != fingerprint:
                raise ValueError("Unexpected pre-existing shard")
            stats = smooth_audit(payload)
        else:
            program = make_smooth_jog_program(batch, queries, seed=config["seed"],
                episode_offset=offset+index*batch, period=sim.period)
            sim.rng.manual_seed(rollout_seed)
            q, v, command, mode, fell, fall_time = sim.rollout(queries+1,
                command_schedule=program["commands"], initial_phase=program["initial_phase"],
                progress_callback=lambda elapsed: print(
                    f"{phase} batch={index+1}/{batches} simulated={elapsed:.0f}/{config['duration_s']:.0f}s", flush=True))
            torch.cuda.synchronize()
            payload = dict(q_log=q.cpu().numpy(), v_log=v.cpu().numpy(), cmd_log=command.cpu().numpy(),
                mode_log=mode.cpu().numpy(), mode_names=np.asarray(MODE_NAMES),
                time_log=np.broadcast_to(np.arange(queries+1)*.02, (batch, queries+1)),
                fell=fell.cpu().numpy(), fall_time=fall_time.cpu().numpy(), sim_dt=sim.sim_dt,
                control_dt=sim.control_dt, log_dt=sim.log_dt, policy=sim.policy_name,
                model="g1_29dof.xml", batch_size=np.int64(batch), episode_ids=program["episode_ids"],
                trajectory_ids=np.asarray([f"{config['collection_id']}:{identity}" for identity in program["episode_ids"]]),
                command_profile_labels=program["profile_labels"], command_profile_variants=program["profile_variants"],
                command_profile_parameters=program["profile_parameters"],
                command_behavior_log=program["behavior"], behavior_names=np.asarray(BEHAVIOR_NAMES),
                planned_cmd_log=program["commands"], initial_phase_step=program["initial_phase"],
                command_integrated_reference_log=program["command_integrated_reference"],
                path_core_start_query=program["path_core_start_query"], path_core_end_query=program["path_core_end_query"],
                path_period_s=program["path_period_s"], rollout_seed=np.int64(rollout_seed),
                collection_fingerprint=np.asarray(fingerprint),
                coordinate_conventions=np.asarray(json.dumps(dict(q_position="world", q_orientation="world_wxyz",
                    v_linear="world", v_angular="body", reference_pose="initial ground/yaw frame; integrated commanded unicycle",
                    processing_frame="fixed ground/yaw frame per window using utils/relative_frame.py",
                    policy_memory="unchanged between commands; reset only at episode start",
                    split_owner="KCF_MPC processing pipeline; no splits assigned by collector"))),
                **sim.full_log_extras())
            stats = smooth_audit(payload, sim.policy)
            compressed_archive(archive_path, payload)
            del program, q, v, command, mode, fell, fall_time
        entry = dict(filename=filename, sha256=digest(archive_path), bytes=archive_path.stat().st_size,
                     elapsed_seconds=time.monotonic()-started, audit=stats)
        atomic_json(archive_path.with_suffix(".json"), entry)
        manifest["shards"].append(entry)
        manifest.update(totals=totals(manifest["shards"]), bytes=sum(item["bytes"] for item in manifest["shards"]),
                        elapsed_seconds=sum(item["elapsed_seconds"] for item in manifest["shards"]))
        manifest["estimated_remaining_seconds"] = manifest["elapsed_seconds"]/(index+1)*(batches-index-1)
        atomic_json(path, manifest)
        print(f"{phase} shard={index+1}/{batches} valid={stats['valid_transitions']}/{stats['queries']} failed_episodes={stats['failed_episodes']}/{batch}", flush=True)
        del payload
        sim.policy_log.records.clear()
        gc.collect()
    manifest["quality"] = quality(manifest["totals"], config, phase)
    manifest["status"] = "complete" if manifest["quality"]["passed"] else "quality_failed"
    atomic_json(path, manifest)
    del sim
    gc.collect()
    if not manifest["quality"]["passed"]:
        raise RuntimeError(f"Smooth {phase} quality failed: {manifest['quality']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"sim/parallel/jog_smooth_collection_v1.json")
    parser.add_argument("--phase", choices=["pilot", "full", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = ROOT/config["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"collection.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 0 is unavailable in this environment")
    torch.set_num_threads(config["policy_threads"])
    branch = subprocess.check_output(["git", "-C", str(ROOT/"KCF_MPC"), "branch", "--show-current"], text=True).strip()
    if branch != "integrated-pose-runtime":
        raise ValueError("KCF_MPC must remain on integrated-pose-runtime")
    sources = ["utils/jog_commands.py", "utils/jog_smooth_commands.py", "utils/jog_logging.py", "utils/relative_frame.py",
               "policy/jog/jog_actor.py", "sim/parallel/parallel_sim_jog.py", "scripts/collect_jog_dataset.py",
               "scripts/collect_smooth_jog_dataset.py"]
    provenance = dict(kcf_branch=branch,
        kcf_commit=subprocess.check_output(["git", "-C", str(ROOT/"KCF_MPC"), "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={name:digest(ROOT/name) for name in sources},
        model_sha256=digest(ROOT/"models/g1_29dof/g1_29dof.xml"),
        policy_sha256=digest(ROOT/"policy/jog/g1_29dof_jog_lstm_v0_cont.onnx"),
        gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch_version=torch.__version__, split_owner="KCF_MPC processing pipeline")
    atomic_json(root/"run_status.json", dict(status="running", phase=args.phase, config=config, provenance=provenance))
    try:
        pilot = run_phase(config, root, "pilot", provenance)
        phase = "pilot"
        result = pilot
        if args.phase in ("full", "all"):
            phase, result = "full", run_phase(config, root, "full", provenance)
        atomic_json(root/"run_status.json", dict(status="complete" if phase == "full" else "pilot_complete",
            phase=phase, config=config, provenance=provenance, totals=result["totals"], quality=result["quality"]))
    except BaseException as error:
        atomic_json(root/"run_status.json", dict(status="failed", phase=args.phase, config=config, error=str(error)))
        raise


if __name__ == "__main__":
    main()
