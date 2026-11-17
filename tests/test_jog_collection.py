"""Program coverage, batching invariance, and scheduled policy-memory behavior."""

import numpy as np
import pytest

from utils.jog_commands import (BEHAVIOR_NAMES, PROFILE_NAMES, command_behavior,
                                make_jog_program, validate_commands)
from utils.relative_frame import RelativeFrame
from scripts.collect_jog_dataset import compressed_archive, valid_pose


def test_programs_are_reproducible_independently_of_batch_boundaries():
    whole = make_jog_program(80, 1000, seed=20261114, episode_offset=17)
    left = make_jog_program(31, 1000, seed=20261114, episode_offset=17)
    right = make_jog_program(49, 1000, seed=20261114, episode_offset=48)
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate((left[key], right[key])))
    assert set(whole["profile_labels"]) == set(PROFILE_NAMES)
    np.testing.assert_array_equal(whole["commands"][:, -150:], 0.)
    assert (whole["initial_phase"][(whole["episode_ids"] % 5) == 0] == 0).all()


def test_full_design_covers_every_directed_behavior_pair_and_trained_boundary():
    p = make_jog_program(10240, 1000, seed=20261114)
    validate_commands(p["commands"])
    assert set(np.unique(p["behavior"])) == set(range(len(BEHAVIOR_NAMES)))
    a, b = p["behavior"][:, :-1], p["behavior"][:, 1:]
    changed = a != b
    counts = np.zeros((7, 7), int)
    np.add.at(counts, (a[changed], b[changed]), 1)
    assert counts[~np.eye(7, dtype=bool)].min() >= 50
    for endpoint in [.5, 1.5, -.5, -1.]:
        assert np.any(p["commands"][..., 0] == endpoint)
    for endpoint in [1., 2., -1., -2.]:
        assert np.any(p["commands"][..., 2] == endpoint)
    assert set(p["initial_phase"]) == set(range(43))


@pytest.mark.parametrize("command", [[.1, 0, 0], [0, .1, 0], [.8, 0, .8], [0, 0, .5], [1.6, 0, 0]])
def test_unsupported_command_gaps_are_rejected(command):
    with pytest.raises(ValueError):
        validate_commands(np.array([[command]], dtype=np.float32))


def test_scheduled_transitions_preserve_policy_memory_and_override_random_clock(monkeypatch):
    from test_jog_logging import cpu_parallel, check_policy_records
    sim = cpu_parallel(monkeypatch, 50)
    commands = np.array([[[0, 0, 0], [.8, 0, 0], [0, 0, 0], [0, 0, -1.2]]]*2, dtype=np.float32)
    initial_phase = np.array([41, 8])
    sim.rollout(5, command_schedule=commands, initial_phase=initial_phase)
    data = sim.full_log_extras()
    np.testing.assert_array_equal(data["policy_cmd_log"], commands)
    np.testing.assert_array_equal(data["policy_phase_step_log"][:, 0], initial_phase)
    assert np.any(data["policy_h_in_log"][:, 2] != 0.)
    check_policy_records(data, sim.policy)


def test_relative_frame_is_ground_anchored_and_handles_random_global_origins_and_yaws():
    origin = np.array([4., -2., 0.])
    yaw = 2.3
    frame = RelativeFrame.from_pose(origin, [np.cos(yaw/2), 0., 0., np.sin(yaw/2)], yaw_only=True)
    local = np.array([[0., 0., .776], [2., 1., .8], [-1., 3., .75]])
    world = frame.position_to_world(local)
    np.testing.assert_allclose(frame.position_to_local(world), local, atol=1e-12)
    np.testing.assert_allclose(world[:, 2], local[:, 2], atol=1e-12)
    pose = np.zeros((3, 36))
    pose[:, :3] = world
    pose[:, 3:7] = frame.orientation_world
    assert valid_pose(pose).all()
    pose[1, 2] = .3
    assert not valid_pose(pose)[1]


def test_compressed_shards_roundtrip_without_pickle_or_partial_files(tmp_path):
    path = tmp_path/"batch.npz"
    data = dict(q=np.arange(72, dtype=np.float32).reshape(2, 36),
                labels=np.array(["start_stop", "standing"]),
                metadata=np.array('{"frame":"world"}'))
    compressed_archive(path, data)
    with np.load(path, allow_pickle=False) as archive:
        for key in data:
            np.testing.assert_array_equal(archive[key], data[key])
    assert not list(tmp_path.glob("*.partial"))
