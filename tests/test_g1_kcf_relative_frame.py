"""Compatibility checks for relative coordinates in the existing 84-D state."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

from utils.g1_kcf_state_frame import G1KCFRelativeFrame
from utils.relative_frame import RelativeFrame


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_PATH = (
    ROOT / "data/data/sync_lissa_kcf02/g1_23dof_data.npz"
)
POLICY_COMMANDS_PATH = (
    ROOT / "data/data/sync_lissa_kcf02/applied_policy_commands.npz"
)
RAW_WINDOW_START = 100
WINDOW_STATES = 25


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _build_runtime_states(
    q: np.ndarray,
    v: np.ndarray,
    times: np.ndarray,
    previous_actions: np.ndarray,
) -> np.ndarray:
    """Call the production runtime state builder without changing its module."""
    pytest.importorskip("torch")
    single_sim_dir = str(ROOT / "sim/single")
    if single_sim_dir not in sys.path:
        sys.path.insert(0, single_sim_dir)
    from control_base import RobotState
    from utils.g1_kcf_state import g1_closed_loop_state
    from policy.config import G1Config

    gait_period = float(G1Config().gait_period)
    theta_previous = None
    result = []
    for time_value, q_value, v_value, previous_action in zip(
        times, q, v, previous_actions
    ):
        robot_state = RobotState(
            t=float(time_value),
            q=q_value,
            v=v_value,
            prev_action=previous_action,
        )
        state = g1_closed_loop_state(
            robot_state,
            gait_period,
            theta_prev=theta_previous,
        )
        result.append(state)
        theta_previous = float(state[2])
    return np.asarray(result)


def _load_recorded_window():
    if not TRAJECTORY_PATH.is_file() or not POLICY_COMMANDS_PATH.is_file():
        pytest.skip("recorded G1 trajectory and policy actions are required")

    with np.load(TRAJECTORY_PATH, allow_pickle=False) as archive:
        q_log = np.asarray(archive["q_log"][0])
        v_log = np.asarray(archive["v_log"][0])
        log_dt = float(archive["log_dt"])
        control_dt = float(archive["control_dt"])
    with np.load(POLICY_COMMANDS_PATH, allow_pickle=False) as archive:
        policy_times = np.asarray(archive["time"])
        previous_actions = np.asarray(archive["previous_policy_action"])

    stride_float = control_dt / log_dt
    stride = int(round(stride_float))
    if not np.isclose(stride_float, stride, rtol=0.0, atol=1e-12):
        raise AssertionError("KCF control period is not an integer log stride")
    if RAW_WINDOW_START % stride:
        raise AssertionError("raw window start must coincide with a policy tick")

    raw_indices = RAW_WINDOW_START + stride * np.arange(WINDOW_STATES)
    policy_indices = raw_indices // stride
    times = raw_indices.astype(np.float64) * log_dt
    np.testing.assert_allclose(
        policy_times[policy_indices], times, rtol=0.0, atol=2e-12
    )
    return (
        q_log[raw_indices],
        v_log[raw_indices],
        times,
        previous_actions[policy_indices],
    )


def test_unwrapped_heading_round_trip_preserves_continuous_turn_count():
    anchor = np.zeros(84, dtype=np.float64)
    anchor[:3] = [2.0, -3.0, 4.0 * np.pi + 0.3]
    states = np.stack((anchor, anchor.copy()))
    states[1, :3] = [2.5, -2.8, anchor[2] + 2.0 * np.pi + 0.2]

    transform = G1KCFRelativeFrame.from_anchor(
        anchor, _yaw_quaternion(0.3)
    )
    local = transform.state_to_local(states)

    np.testing.assert_allclose(local[0, :3], 0.0, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(local[1, 2], 2.0 * np.pi + 0.2, atol=2e-15)
    np.testing.assert_allclose(
        transform.state_to_world(local), states, rtol=0.0, atol=2e-15
    )


def test_planar_state_constructor_and_pose_batch_share_state_transform():
    anchor = np.zeros(84, dtype=np.float64)
    anchor[:3] = [3.0, -2.0, 2.0 * np.pi + 0.4]
    poses = np.array(
        [
            anchor[:3],
            [3.5, -1.7, anchor[2] + 0.2],
            [2.2, -2.4, anchor[2] - 0.3],
        ]
    )
    states = np.broadcast_to(anchor, (len(poses), 84)).copy()
    states[:, :3] = poses

    transform = G1KCFRelativeFrame.from_planar_state(anchor)
    local_poses = transform.planar_pose_to_local(poses)

    np.testing.assert_allclose(local_poses[0], 0.0, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        local_poses, transform.state_to_local(states)[:, :3],
        rtol=0.0, atol=2e-15,
    )
    np.testing.assert_allclose(
        transform.planar_pose_to_world(local_poses), poses,
        rtol=0.0, atol=2e-15,
    )


def test_recorded_84d_window_round_trip_and_global_pose_invariance():
    q_a, v_a, times, previous_actions = _load_recorded_window()
    states_a = _build_runtime_states(q_a, v_a, times, previous_actions)
    original_states_a = states_a.copy()
    transform_a = G1KCFRelativeFrame.from_anchor(
        states_a[0], q_a[0, 3:7]
    )

    local_a = transform_a.state_to_local(states_a)
    reconstructed_a = transform_a.state_to_world(local_a)

    assert states_a.shape == (WINDOW_STATES, 84)
    assert transform_a.transformed_indices == (0, 1, 2)
    np.testing.assert_array_equal(states_a, original_states_a)
    np.testing.assert_allclose(local_a[0, :3], 0.0, rtol=0.0, atol=2e-7)
    np.testing.assert_array_equal(local_a[:, 3:], states_a[:, 3:])
    np.testing.assert_allclose(
        reconstructed_a, states_a, rtol=0.0, atol=5e-7
    )

    # Build the physically equivalent trajectory at (10, 20) with +90 deg
    # world yaw.  RelativeFrame supplies every rigid transformation; the
    # body's angular velocity, joints, and controller memory remain intrinsic.
    global_transform = RelativeFrame(
        np.array([10.0, 20.0, 0.0]), _yaw_quaternion(0.5 * np.pi)
    )
    q_b = q_a.copy()
    v_b = v_a.copy()
    q_b[:, :3] = global_transform.position_to_world(q_a[:, :3])
    q_b[:, 3:7] = global_transform.orientation_to_world(q_a[:, 3:7])
    v_b[:, :3] = global_transform.linear_velocity_to_world(v_a[:, :3])
    states_b = _build_runtime_states(q_b, v_b, times, previous_actions)
    transform_b = G1KCFRelativeFrame.from_anchor(
        states_b[0], q_b[0, 3:7]
    )
    local_b = transform_b.state_to_local(states_b)

    np.testing.assert_allclose(local_b[0, :3], 0.0, rtol=0.0, atol=2e-6)
    np.testing.assert_allclose(local_b, local_a, rtol=0.0, atol=3e-6)
    np.testing.assert_array_equal(local_b[:, 3:], states_b[:, 3:])
    np.testing.assert_allclose(
        transform_b.state_to_world(local_b), states_b, rtol=0.0, atol=2e-6
    )
