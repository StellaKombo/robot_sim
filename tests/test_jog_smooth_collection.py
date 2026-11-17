"""Feasible geometric paths, continuous excitation, and persistent actor state."""

import numpy as np
import pytest

from utils.jog_commands import validate_commands
from utils.jog_smooth_commands import (PROFILE_NAMES, _periodic_path,
    integrate_unicycle, make_smooth_jog_program)


def test_smooth_program_is_independent_of_batch_boundaries():
    whole = make_smooth_jog_program(40, 3000, seed=2026111401)
    left = make_smooth_jog_program(17, 3000, seed=2026111401)
    right = make_smooth_jog_program(23, 3000, seed=2026111401, episode_offset=2_000_017)
    for key in whole:
        np.testing.assert_array_equal(whole[key], np.concatenate((left[key], right[key])))
    assert set(whole["profile_labels"]) == set(PROFILE_NAMES)


@pytest.mark.parametrize("kind", ["figure_eight", "ovals"])
@pytest.mark.parametrize("backward", [False, True])
def test_periodic_paths_close_and_have_the_expected_heading_winding(kind, backward):
    sample, metadata = _periodic_path(np.random.default_rng(15), kind, 53.5, backward)
    period = metadata["period_s"]
    dt = period/24000
    commands = sample((np.arange(24000)+.5)*dt)
    validate_commands(commands[None])
    path = integrate_unicycle(commands, dt)
    assert np.linalg.norm(path[-1, :2]) < .0003
    if kind == "figure_eight":
        assert abs(path[-1, 2]) < .0001
        assert commands[:, 2].min() < -.1 and commands[:, 2].max() > .1
    else:
        assert abs(abs(path[-1, 2])-2*np.pi) < .0001
    assert np.max(np.abs(commands[:, 2])) < .48
    np.testing.assert_allclose(sample(np.array([0.])), sample(np.array([period])), atol=1e-10)


def test_full_supplement_covers_smooth_joint_excitation_and_complete_paths():
    program = make_smooth_jog_program(1536, 3000, seed=2026111401)
    command = program["commands"]
    validate_commands(command)
    np.testing.assert_array_equal(command[:, -150:], 0.)
    assert set(program["initial_phase"]) == set(range(43))
    delta = np.abs(np.diff(command, axis=1))
    drive = command[:, 1:, 0]*command[:, :-1, 0] > 0.
    pivot = ((command[:, 1:, 0] == 0.) & (command[:, :-1, 0] == 0.)
             & (command[:, 1:, 2]*command[:, :-1, 2] > 0.))
    assert np.max(delta[drive | pivot]) < .06
    joint = drive & (delta[..., 0] > 1e-7) & (delta[..., 2] > 1e-7)
    assert joint.sum() > .5*1536*3000
    for profile in ("figure_eight", "ovals"):
        selected = program["profile_labels"] == profile
        available = (program["path_core_end_query"][selected]-program["path_core_start_query"][selected])*.02
        assert np.all(program["path_period_s"][selected] < available)
        assert np.any(np.char.startswith(program["profile_variants"][selected], "backward"))
    assert np.any((command[..., 0] != 0.) & (np.abs(command[..., 2]) < .01) & (command[..., 2] != 0.))


def test_rapid_stops_are_retained_between_smooth_segments():
    program = make_smooth_jog_program(100, 3000, seed=2026111401)
    short_stops = 0
    for row in np.flatnonzero(program["profile_labels"] == "smooth_restart"):
        moving = np.any(program["commands"][row] != 0., axis=-1)
        stops = np.flatnonzero(moving[:-1] & ~moving[1:])+1
        starts = np.flatnonzero(~moving[:-1] & moving[1:])+1
        for stop in stops:
            following = starts[starts > stop]
            if len(following) and 4 <= following[0]-stop <= 25:
                short_stops += 1
    assert short_stops > 0


def test_held_unicycle_integration_matches_a_known_circle():
    command = np.broadcast_to([1., 0., .25], (2000, 3))
    path = integrate_unicycle(command, .02)
    time = np.arange(2001)*.02
    np.testing.assert_allclose(path[:, 0], 4*np.sin(.25*time), atol=1e-11)
    np.testing.assert_allclose(path[:, 1], 4*(1-np.cos(.25*time)), atol=1e-11)
    np.testing.assert_allclose(path[:, 2], .25*time, atol=1e-11)


def test_smooth_inputs_preserve_logged_onnx_memory_and_policy_queries(monkeypatch):
    from test_jog_logging import cpu_parallel, check_policy_records
    sim = cpu_parallel(monkeypatch, 50)
    program = make_smooth_jog_program(20, 3000, seed=2026111401)
    rows = [np.flatnonzero(program["profile_labels"] == name)[0] for name in ("slalom", "figure_eight")]
    command = program["commands"][rows, :180]
    sim.rollout(181, command_schedule=command, initial_phase=program["initial_phase"][rows])
    data = sim.full_log_extras()
    np.testing.assert_array_equal(data["policy_cmd_log"], command)
    assert np.any(data["policy_h_in_log"][:, 100] != 0.)
    check_policy_records(data, sim.policy)
