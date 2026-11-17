"""Collect and audit resumable jogging trajectories; KCF owns dataset splitting."""

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
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ["ROBOT_SIM_ROOT_DIR"] = str(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sim.parallel.parallel_sim_jog import JogSimConfig, ParallelJogSim, MODE_NAMES
from utils.jog_commands import BEHAVIOR_NAMES, PROFILE_NAMES, make_jog_program
from utils.math_utils import world_to_body
from utils.relative_frame import RelativeFrame


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def compressed_archive(path, arrays):
    temporary = path.with_name(path.name + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for name, value in arrays.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


def valid_pose(q):
    qw, qx, qy, qz = np.moveaxis(q[..., 3:7], -1, 0)
    roll = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx*qx+qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy-qz*qx), -1., 1.))
    return (np.isfinite(q).all(-1) & (q[..., 2] >= .5)
            & (np.abs(roll) <= .8) & (np.abs(pitch) <= .8))


def audit(data, policy=None):
    """Check all recorded state chains; counts describe valid motion prefixes."""
    q, v = data["policy_q_log"], data["policy_v_log"]
    command, mode = data["policy_cmd_log"], data["policy_mode_log"]
    batch, queries, _ = q.shape
    for name, value in data.items():
        if name.startswith("policy_") and name.endswith("_log") and np.issubdtype(value.dtype, np.floating):
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite {name}")
    np.testing.assert_array_equal(q, data["q_log"][:, :-1])
    np.testing.assert_array_equal(v, data["v_log"][:, :-1])
    np.testing.assert_array_equal(data["terminal_q"], data["q_log"][:, -1])
    np.testing.assert_array_equal(command, data["cmd_log"][:, 1:])
    np.testing.assert_array_equal(command, data["planned_cmd_log"])
    np.testing.assert_array_equal(data["policy_prev_action_log"][:, 1:], data["policy_action_log"][:, :-1])
    np.testing.assert_array_equal(data["terminal_prev_action"], data["policy_action_log"][:, -1])
    np.testing.assert_allclose(np.diff(data["policy_time_log"], axis=1), .02, rtol=0., atol=1e-10)
    np.testing.assert_array_equal(data["policy_phase_step_log"][:, 0], data["initial_phase_step"])
    np.testing.assert_array_equal(data["policy_phase_step_log"][:, 1:],
                                  (data["policy_phase_step_log"][:, :-1]+1) % 43)
    np.testing.assert_array_equal(data["policy_obs_log"][..., 61:90], data["policy_prev_action_log"])
    np.testing.assert_array_equal(data["policy_obs_log"][..., 90:93], command)
    moving = np.any(command != 0., axis=-1)
    np.testing.assert_array_equal(data["policy_gait_phase_log"][~moving], 0.)
    np.testing.assert_allclose(data["policy_gait_phase_log"][moving],
                               data["policy_clock_log"][moving], atol=1e-6, rtol=0.)
    assert data["terminal_control_aligned"].all()
    for name in data["policy_state_names"].tolist():
        incoming, outgoing = f"policy_{name}_log", f"policy_{name[:-3]}_out_log"
        np.testing.assert_array_equal(data[incoming][:, 0], 0.)
        np.testing.assert_array_equal(data[outgoing][:, :-1], data[incoming][:, 1:])
        np.testing.assert_array_equal(data[outgoing][:, -1], data[f"terminal_{name}"])
    if policy is not None:
        for index in sorted({0, queries//4, queries//2, queries-1}):
            policy.state = {name: np.moveaxis(data[f"policy_{name}_log"][:, index], 0, 1).copy()
                            for name in policy.state_shapes}
            result = policy(torch.from_numpy(data["policy_obs_log"][:, index].copy())).numpy()
            np.testing.assert_array_equal(result, data["policy_action_log"][:, index])

    # The same ground/yaw frame used by KCF must preserve physical coordinates.
    for index in range(min(4, batch)):
        anchor = q[index, 0]
        frame = RelativeFrame.from_pose([anchor[0], anchor[1], 0.], anchor[3:7], yaw_only=True)
        local = frame.position_to_local(q[index, :, :3])
        np.testing.assert_allclose(frame.position_to_world(local), q[index, :, :3], atol=2e-5)
        np.testing.assert_allclose(local[0, :2], 0., atol=2e-5)
        np.testing.assert_allclose(local[:, 2], q[index, :, 2], atol=2e-5)

    healthy = valid_pose(q)
    alive = np.logical_and.accumulate(healthy, axis=1)
    following = np.concatenate((healthy[:, 1:], valid_pose(data["terminal_q"])[:, None]), axis=1)
    usable = alive & following
    data["valid_transition_mask"] = usable
    first_failure = np.where(usable.all(1), -1, np.argmin(usable, axis=1)).astype(np.int64)
    data["first_invalid_transition"] = first_failure
    behavior = data["command_behavior_log"]
    matrix = np.zeros((7, 7), dtype=np.int64)
    changes = (behavior[:, 1:] != behavior[:, :-1]) & usable[:, 1:] & usable[:, :-1]
    np.add.at(matrix, (behavior[:, :-1][changes], behavior[:, 1:][changes]), 1)
    coarse = np.zeros((4, 4), dtype=np.int64)
    changed = (mode[:, 1:] != mode[:, :-1]) & usable[:, 1:] & usable[:, :-1]
    np.add.at(coarse, (mode[:, :-1][changed], mode[:, 1:][changed]), 1)
    velocity = world_to_body(q[..., 3:7], v[..., :3])
    settled = (np.linalg.norm(velocity[..., :2], axis=-1) < .1) & (np.abs(v[..., 5]) < .1)
    starts = moving[:, 1:] & ~moving[:, :-1] & usable[:, 1:]
    stop_events = ~moving[:, 1:] & moving[:, :-1] & usable[:, 1:]
    phases = data["policy_phase_step_log"][:, 1:][starts]
    long_stops = 0
    for index in range(batch):
        zero = (~moving[index] & usable[index]).astype(np.int8)
        edges = np.diff(np.r_[0, zero, 0])
        long_stops += int(np.sum(np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1) >= 150))
    return dict(
        episodes=batch, queries=batch*queries, valid_transitions=int(usable.sum()),
        failed_episodes=int((first_failure >= 0).sum()), height_falls=int(data["fell"].sum()),
        profiles=dict(Counter(data["command_profile_labels"].tolist())),
        behaviors=np.bincount(behavior[usable], minlength=7).tolist(),
        mode_transitions=coarse.tolist(), behavior_transitions=matrix.tolist(),
        starts=int(starts.sum()), stops=int(stop_events.sum()),
        unsettled_restarts=int((starts & ~settled[:, 1:]).sum()),
        settled_zero_samples=int((~moving & usable & settled).sum()), long_zero_holds=long_stops,
        start_phase_counts=np.bincount(phases, minlength=43).tolist(),
        initial_phase_counts=np.bincount(data["initial_phase_step"], minlength=43).tolist(),
        command_min=command.min(axis=(0, 1)).tolist(), command_max=command.max(axis=(0, 1)).tolist(),
    )


def aggregate(entries):
    result = dict(episodes=0, queries=0, valid_transitions=0, failed_episodes=0, height_falls=0,
                  starts=0, stops=0, unsettled_restarts=0, settled_zero_samples=0, long_zero_holds=0,
                  profiles={}, behaviors=[0]*7, mode_transitions=np.zeros((4, 4), int).tolist(),
                  behavior_transitions=np.zeros((7, 7), int).tolist(),
                  start_phase_counts=[0]*43, initial_phase_counts=[0]*43)
    vector_keys = ("behaviors", "mode_transitions", "behavior_transitions", "start_phase_counts", "initial_phase_counts")
    for entry in entries:
        stats = entry["audit"]
        for key in result:
            if key == "profiles":
                result[key] = dict(Counter(result[key]) + Counter(stats[key]))
            elif key in vector_keys:
                result[key] = (np.asarray(result[key])+stats[key]).tolist()
            else:
                result[key] += stats[key]
    return result


def quality(stats, config, phase):
    failures = []
    maximum = config["pilot_maximum_failure_fraction"] if phase == "pilot" else config["maximum_failure_fraction"]
    if stats["failed_episodes"] / max(1, stats["episodes"]) > maximum:
        failures.append("too many episodes cross the existing height/tilt envelope")
    if any(stats["profiles"].get(name, 0) == 0 for name in PROFILE_NAMES):
        failures.append("missing motion profiles")
    if min(stats["behaviors"]) == 0:
        failures.append("missing commanded motion behaviors in valid states")
    if min(stats["starts"], stats["stops"], stats["unsettled_restarts"], stats["long_zero_holds"]) == 0:
        failures.append("missing starts, stops, unsettled restarts, or long standing holds")
    if phase == "full":
        counts = np.asarray(stats["behavior_transitions"])[~np.eye(7, dtype=bool)]
        if counts.min() < config["minimum_directed_transition_count"]:
            failures.append("insufficient valid coverage of all 42 directed motion transitions")
        if min(stats["initial_phase_counts"]) == 0 or min(stats["start_phase_counts"]) == 0:
            failures.append("missing initial or restart gait phases")
    return dict(passed=not failures, failures=failures,
                episode_failure_fraction=stats["failed_episodes"]/max(1, stats["episodes"]))


def run_phase(config, root, phase, provenance):
    directory = root / phase
    directory.mkdir(exist_ok=True)
    batch = config["pilot_batch_size"] if phase == "pilot" else config["batch_size"]
    batches = 1 if phase == "pilot" else config["batches"]
    offset_base = 1_000_000 if phase == "pilot" else 0
    fingerprint = hashlib.sha256(json.dumps(dict(config=config, phase=phase, provenance=provenance), sort_keys=True).encode()).hexdigest()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Collection settings or source code changed; use a new output directory")
        for item in manifest["shards"]:
            if digest(directory/item["filename"]) != item["sha256"]:
                raise ValueError(f"Recorded shard checksum changed: {item['filename']}")
    else:
        manifest = dict(status="preparing", phase=phase, fingerprint=fingerprint,
                        config=config, provenance=provenance, shards=[])
    if manifest.get("status") == "complete":
        return manifest
    sim_config = JogSimConfig(
        batch_size=batch, seed=config["seed"],
        xml_path=str(ROOT/"models/g1_29dof/g1_29dof.xml"),
        policy_path=str(ROOT/"policy/jog/g1_29dof_jog_lstm_v0_cont.onnx"),
        log_hz=50, init_joint_noise=config["init_joint_noise"],
        init_yaw_range=config["init_yaw_range"], init_xy_range=config["init_xy_range"],
        policy_threads=config["policy_threads"],
    )
    sim = ParallelJogSim(sim_config)
    queries = int(round(config["duration_s"]/.02))
    manifest.update(status="running", target_episodes=batch*batches, target_shards=batches)
    atomic_json(manifest_path, manifest)
    for index in range(len(manifest["shards"]), batches):
        started = time.monotonic()
        # Reserve one uncompressed batch plus 15 GB; completed shards remain usable.
        if shutil.disk_usage(root).free < config["minimum_free_gb"]*1e9 + batch*queries*3400:
            raise OSError("Insufficient free space to safely write another batch")
        filename = f"g1_29dof_jog_data_{index:05d}.npz"
        path = directory/filename
        rollout_seed = config["seed"] + index + (100_000 if phase == "pilot" else 0)
        if path.exists():
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: archive[name] for name in archive.files}
            if payload["collection_fingerprint"].item() != fingerprint:
                raise ValueError(f"Unexpected pre-existing shard: {path}")
            stats = audit(payload)
        else:
            program = make_jog_program(batch, queries, seed=config["seed"],
                                       episode_offset=offset_base + index*batch, period=sim.period)
            sim.rng.manual_seed(rollout_seed)
            q, v, command, mode, fell, fall_time = sim.rollout(
                queries+1, command_schedule=program["commands"], initial_phase=program["initial_phase"],
                progress_callback=lambda elapsed: print(
                    f"{phase} batch={index+1}/{batches} simulated={elapsed:.0f}/{config['duration_s']:.0f}s", flush=True))
            torch.cuda.synchronize()
            rollout_seconds = time.monotonic()-started
            payload = dict(q_log=q.cpu().numpy(), v_log=v.cpu().numpy(), cmd_log=command.cpu().numpy(),
                           mode_log=mode.cpu().numpy(), mode_names=np.asarray(MODE_NAMES),
                           time_log=np.broadcast_to(np.arange(queries+1)*.02, (batch, queries+1)),
                           fell=fell.cpu().numpy(), fall_time=fall_time.cpu().numpy(),
                           sim_dt=sim.sim_dt, control_dt=sim.control_dt, log_dt=sim.log_dt,
                           policy=sim.policy_name, model="g1_29dof.xml", batch_size=np.int64(batch),
                           episode_ids=program["episode_ids"],
                           trajectory_ids=np.asarray([f"{config['collection_id']}:{i}" for i in program["episode_ids"]]),
                           command_profile_labels=program["profile_labels"],
                           command_profile_variants=program["profile_variants"],
                           command_behavior_log=program["behavior"], behavior_names=np.asarray(BEHAVIOR_NAMES),
                           planned_cmd_log=program["commands"], initial_phase_step=program["initial_phase"],
                           rollout_seed=np.int64(rollout_seed), collection_fingerprint=np.array(fingerprint),
                           coordinate_conventions=np.array(json.dumps(dict(
                               q_position="world", q_orientation="world_wxyz", v_linear="world", v_angular="body",
                               policy_memory="unchanged between commands; reset only at episode start",
                               processing_frame="fixed ground/yaw frame per window using utils/relative_frame.py",
                               split_owner="KCF_MPC processing pipeline; no splits assigned by collector"))),
                           **sim.full_log_extras())
            stats = audit(payload, sim.policy)
            compressed_archive(path, payload)
            print(f"{phase} shard={index+1}/{batches} rollout={rollout_seconds:.1f}s valid={stats['valid_transitions']}/{stats['queries']} failed_episodes={stats['failed_episodes']}/{batch}", flush=True)
            del program, q, v, command, mode, fell, fall_time
        entry = dict(filename=filename, sha256=digest(path), bytes=path.stat().st_size,
                     elapsed_seconds=time.monotonic()-started, audit=stats)
        atomic_json(path.with_suffix(".json"), entry)
        manifest["shards"].append(entry)
        manifest["totals"] = aggregate(manifest["shards"])
        manifest["bytes"] = sum(item["bytes"] for item in manifest["shards"])
        manifest["elapsed_seconds"] = sum(item["elapsed_seconds"] for item in manifest["shards"])
        manifest["estimated_remaining_seconds"] = manifest["elapsed_seconds"]/(index+1)*(batches-index-1)
        atomic_json(manifest_path, manifest)
        del payload
        sim.policy_log.records.clear()
        gc.collect()
    manifest["quality"] = quality(manifest["totals"], config, phase)
    manifest["status"] = "complete" if manifest["quality"]["passed"] else "quality_failed"
    atomic_json(manifest_path, manifest)
    del sim
    gc.collect()
    if not manifest["quality"]["passed"]:
        raise RuntimeError(f"{phase} collection quality checks failed: {manifest['quality']}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"sim/parallel/jog_collection_v1.json")
    parser.add_argument("--phase", choices=["pilot", "full", "all"], default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = ROOT/config["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/"collection.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 0 is unavailable; run outside the restricted sandbox")
    torch.set_num_threads(config["policy_threads"])
    branch = subprocess.check_output(["git", "-C", str(ROOT/"KCF_MPC"), "branch", "--show-current"], text=True).strip()
    if branch != "integrated-pose-runtime":
        raise ValueError("KCF_MPC must use integrated-pose-runtime")
    sources = ["utils/jog_commands.py", "utils/jog_logging.py", "utils/relative_frame.py",
               "policy/jog/jog_actor.py", "sim/parallel/parallel_sim_jog.py", "scripts/collect_jog_dataset.py"]
    provenance = dict(kcf_branch=branch,
                      kcf_commit=subprocess.check_output(["git", "-C", str(ROOT/"KCF_MPC"), "rev-parse", "HEAD"], text=True).strip(),
                      source_sha256={p:digest(ROOT/p) for p in sources},
                      model_sha256=digest(ROOT/"models/g1_29dof/g1_29dof.xml"),
                      policy_sha256=digest(ROOT/"policy/jog/g1_29dof_jog_lstm_v0_cont.onnx"),
                      gpu=torch.cuda.get_device_name(0), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      torch_version=torch.__version__, split_owner="KCF_MPC processing pipeline")
    atomic_json(root/"run_status.json", dict(status="running", phase=args.phase, config=config, provenance=provenance))
    try:
        pilot = run_phase(config, root, "pilot", provenance)
        if args.phase in ("full", "all"):
            full = run_phase(config, root, "full", provenance)
            atomic_json(root/"run_status.json", dict(status="complete", config=config, provenance=provenance,
                        full_manifest=str(root/"full/manifest.json"), totals=full["totals"], quality=full["quality"]))
        else:
            atomic_json(root/"run_status.json", dict(status="pilot_complete", config=config,
                        pilot_manifest=str(root/"pilot/manifest.json"), totals=pilot["totals"], quality=pilot["quality"]))
    except BaseException as error:
        atomic_json(root/"run_status.json", dict(status="failed", error=str(error), phase=args.phase, config=config))
        raise


if __name__ == "__main__":
    main()
