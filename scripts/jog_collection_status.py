"""Read jogging collection progress without initializing CUDA or loading datasets."""

import argparse
from collections import deque
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1]/"data/data/jog_kcf_collection_v1")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--phase", choices=["auto", "pilot", "full"], default="auto")
    args = parser.parse_args()
    root = args.directory
    run = json.loads((root/"run_status.json").read_text())
    phase = args.phase
    if phase == "auto":
        phase = "full" if (root/"full/manifest.json").exists() else "pilot"
    manifest_path = root/phase/"manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    totals = manifest.get("totals", {})
    matrix = totals.get("behavior_transitions", [[0]*7 for _ in range(7)])
    directed = [matrix[i][j] for i in range(7) for j in range(7) if i != j]
    result = dict(status=run["status"], phase=phase,
                  completed_shards=len(manifest.get("shards", [])), target_shards=manifest.get("target_shards"),
                  episodes=totals.get("episodes", 0), target_episodes=manifest.get("target_episodes"),
                  valid_transitions=totals.get("valid_transitions", 0),
                  recorded_transitions=totals.get("queries", 0),
                  failed_episodes=totals.get("failed_episodes", 0),
                  size_gb=round(manifest.get("bytes", 0)/1e9, 3),
                  remaining_minutes=(round(manifest["estimated_remaining_seconds"]/60, 1)
                                     if "estimated_remaining_seconds" in manifest else None),
                  directed_pairs_covered=sum(count > 0 for count in directed),
                  minimum_directed_pair_examples=min(directed),
                  starts=totals.get("starts", 0), stops=totals.get("stops", 0),
                  unsettled_restarts=totals.get("unsettled_restarts", 0),
                  profiles=totals.get("profiles", {}),
                  quality=manifest.get("quality"), error=run.get("error"))
    if "_smooth_" in run.get("config", {}).get("collection_id", ""):
        result.update(varying_same_mode_pairs=totals.get("varying_same_mode_pairs", 0),
                      jointly_varying_drive_pairs=totals.get("jointly_varying_drive_pairs", 0),
                      complete_valid_reference_loops=totals.get("complete_valid_reference_loops", {}))
    log = root/("pilot.log" if phase == "pilot" and (root/"pilot.log").exists() else "collection.log")
    if log.exists():
        with log.open() as handle:
            lines = list(deque(handle, maxlen=2))
        result["latest_log"] = [line.strip() for line in lines]
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Status: {result['status']}; shards {result['completed_shards']}/{result['target_shards']}")
        print(f"Episodes: {result['episodes']}/{result['target_episodes']}; "
              f"valid transitions: {result['valid_transitions']:,}/{result['recorded_transitions']:,}")
        remaining = "waiting for first batch" if result["remaining_minutes"] is None else f"{result['remaining_minutes']} min"
        print(f"Failed episodes: {result['failed_episodes']}; saved: {result['size_gb']} GB; remaining estimate: {remaining}")
        if "varying_same_mode_pairs" in result:
            print(f"Varying input pairs: {result['varying_same_mode_pairs']:,}; "
                  f"simultaneous speed/yaw pairs: {result['jointly_varying_drive_pairs']:,}")
            print(f"Complete valid reference cycles: {result['complete_valid_reference_loops']}")
        else:
            print(f"Directed motion pairs: {result['directed_pairs_covered']}/42; "
                  f"minimum examples: {result['minimum_directed_pair_examples']}")
        print(f"Starts/stops: {result['starts']}/{result['stops']}; "
              f"unsettled restarts: {result['unsettled_restarts']}")
        for line in result.get("latest_log", []):
            print(line)
        if result["error"]:
            print("Error:", result["error"])


if __name__ == "__main__":
    main()
