"""Physical translation/yaw symmetry validation for the G1 MuJoCo system."""

from __future__ import annotations

import math

import numpy as np
import pytest


from utils.relative_frame import RelativeFrame

try:
    import mujoco
    import torch
except ModuleNotFoundError:
    mujoco = None
    torch = None
else:
    from policy.config import G1Config
    from sim.single.simulation import POLICY_PATH, SIM_DT, XML_PATH, Simulation


requires_simulation_environment = pytest.mark.skipif(
    mujoco is None or torch is None,
    reason="the robot_sim environment with MuJoCo and PyTorch is required",
)


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array(
        [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
        dtype=np.float64,
    )


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _rotation_z(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[cosine, -sine, 0.0],
         [sine, cosine, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _command(control_index: int) -> np.ndarray:
    """Deterministic body-frame command sequence shared by both simulations."""
    if control_index < 15:
        return np.array([0.35, 0.10, 0.15], dtype=np.float32)
    if control_index < 30:
        return np.array([0.55, -0.20, -0.10], dtype=np.float32)
    if control_index < 45:
        return np.array([0.20, 0.00, 0.30], dtype=np.float32)
    return np.array([0.40, 0.15, -0.20], dtype=np.float32)


def _initialize_equivalent_state(
    simulation: Simulation,
    *,
    translation: np.ndarray,
    yaw: float,
) -> RelativeFrame:
    simulation.reset()
    rotation = _rotation_z(yaw)
    yaw_quaternion = _yaw_quaternion(yaw)
    initial_position = np.asarray(simulation.cfg.default_base_pos[:3])
    initial_orientation = np.asarray(simulation.cfg.default_base_pos[3:7])

    simulation.data.qpos[:3] = rotation @ initial_position + translation
    simulation.data.qpos[3:7] = _quaternion_multiply(
        yaw_quaternion, initial_orientation
    )
    # MuJoCo free-joint linear velocity is world-frame, while its angular
    # velocity is already body-frame.
    simulation.data.qvel[:3] = rotation @ np.array([0.08, -0.03, 0.01])
    simulation.data.qvel[3:6] = np.array([0.02, -0.01, 0.04])
    simulation.data.qvel[6:] = 0.0
    simulation.data.time = 0.0
    simulation.prev_action[:] = 0.0
    simulation.action[:] = 0.0
    simulation.data.ctrl[:] = 0.0
    simulation.n_steps = 0
    mujoco.mj_forward(simulation.model, simulation.data)

    ground_origin = np.array(
        [simulation.data.qpos[0], simulation.data.qpos[1], 0.0]
    )
    return RelativeFrame.from_pose(
        ground_origin, simulation.data.qpos[3:7], yaw_only=True
    )


def _touch_values(simulation: Simulation) -> np.ndarray:
    return np.array(
        [
            simulation.data.sensordata[
                simulation.model.sensor(name).adr[0]
            ]
            for name in ("left_foot_touch", "right_foot_touch")
        ]
    )


def _orientation_angle_error(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    alignment = np.clip(
        np.abs(np.sum(left * right, axis=-1)), 0.0, 1.0
    )
    return 2.0 * np.arccos(alignment)


def _run_equivalent_rollouts(duration: float = 1.2):
    config = G1Config()
    simulation_a = Simulation(
        XML_PATH, POLICY_PATH, SIM_DT, config, headless=True
    )
    simulation_b = Simulation(
        XML_PATH, POLICY_PATH, SIM_DT, config, headless=True
    )
    frame_a = _initialize_equivalent_state(
        simulation_a, translation=np.zeros(3), yaw=0.0
    )
    frame_b = _initialize_equivalent_state(
        simulation_b,
        translation=np.array([10.0, 20.0, 0.0]),
        yaw=0.5 * np.pi,
    )

    q_a, q_b, v_a, v_b = [], [], [], []
    contacts_a, contacts_b = [], []
    observations_a, observations_b = [], []
    actions_a, actions_b = [], []

    def record_state():
        q_a.append(simulation_a.data.qpos.copy())
        q_b.append(simulation_b.data.qpos.copy())
        v_a.append(simulation_a.data.qvel.copy())
        v_b.append(simulation_b.data.qvel.copy())
        contacts_a.append(_touch_values(simulation_a))
        contacts_b.append(_touch_values(simulation_b))

    record_state()
    physics_steps = int(round(duration / SIM_DT))
    control_index = 0
    for step in range(physics_steps):
        if step % simulation_a.decimation == 0:
            command = _command(control_index)
            simulation_a.set_cmd(command)
            simulation_b.set_cmd(command)
            observations_a.append(
                simulation_a.build_observation().squeeze(0).cpu().numpy()
            )
            observations_b.append(
                simulation_b.build_observation().squeeze(0).cpu().numpy()
            )
            simulation_a.policy_step()
            simulation_b.policy_step()
            actions_a.append(simulation_a.action.copy())
            actions_b.append(simulation_b.action.copy())
            control_index += 1
        simulation_a.step()
        simulation_b.step()
        record_state()

    return {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "q_a": np.asarray(q_a),
        "q_b": np.asarray(q_b),
        "v_a": np.asarray(v_a),
        "v_b": np.asarray(v_b),
        "contacts_a": np.asarray(contacts_a),
        "contacts_b": np.asarray(contacts_b),
        "observations_a": np.asarray(observations_a),
        "observations_b": np.asarray(observations_b),
        "actions_a": np.asarray(actions_a),
        "actions_b": np.asarray(actions_b),
    }


@requires_simulation_environment
def test_g1_closed_loop_dynamics_are_translation_and_yaw_equivariant():
    rollout = _run_equivalent_rollouts()
    frame_a, frame_b = rollout["frame_a"], rollout["frame_b"]
    q_a, q_b = rollout["q_a"], rollout["q_b"]
    v_a, v_b = rollout["v_a"], rollout["v_b"]

    position_a = frame_a.position_to_local(q_a[:, :3])
    position_b = frame_b.position_to_local(q_b[:, :3])
    orientation_a = frame_a.orientation_to_local(q_a[:, 3:7])
    orientation_b = frame_b.orientation_to_local(q_b[:, 3:7])
    linear_velocity_a = frame_a.linear_velocity_to_local(v_a[:, :3])
    linear_velocity_b = frame_b.linear_velocity_to_local(v_b[:, :3])

    position_error = position_a - position_b
    orientation_error = _orientation_angle_error(
        orientation_a, orientation_b
    )
    linear_velocity_error = linear_velocity_a - linear_velocity_b
    angular_velocity_error = v_a[:, 3:6] - v_b[:, 3:6]
    joint_position_error = q_a[:, 7:] - q_b[:, 7:]
    joint_velocity_error = v_a[:, 6:] - v_b[:, 6:]
    observation_error = (
        rollout["observations_a"] - rollout["observations_b"]
    )
    action_error = rollout["actions_a"] - rollout["actions_b"]
    contact_a = rollout["contacts_a"] > 0.0
    contact_b = rollout["contacts_b"] > 0.0
    contact_transition_a = np.diff(contact_a.astype(np.int8), axis=0)
    contact_transition_b = np.diff(contact_b.astype(np.int8), axis=0)
    contact_value_error = rollout["contacts_a"] - rollout["contacts_b"]

    metrics = {
        "position_max": float(np.max(np.abs(position_error))),
        "position_rms": float(np.sqrt(np.mean(position_error**2))),
        "position_final": float(np.linalg.norm(position_error[-1])),
        "orientation_angle_max": float(np.max(orientation_error)),
        "orientation_angle_rms": float(
            np.sqrt(np.mean(orientation_error**2))
        ),
        "orientation_angle_final": float(orientation_error[-1]),
        "linear_velocity_max": float(np.max(np.abs(linear_velocity_error))),
        "linear_velocity_rms": float(
            np.sqrt(np.mean(linear_velocity_error**2))
        ),
        "angular_velocity_max": float(np.max(np.abs(angular_velocity_error))),
        "angular_velocity_rms": float(
            np.sqrt(np.mean(angular_velocity_error**2))
        ),
        "joint_position_max": float(np.max(np.abs(joint_position_error))),
        "joint_position_rms": float(
            np.sqrt(np.mean(joint_position_error**2))
        ),
        "joint_velocity_max": float(np.max(np.abs(joint_velocity_error))),
        "joint_velocity_rms": float(
            np.sqrt(np.mean(joint_velocity_error**2))
        ),
        "observation_max": float(np.max(np.abs(observation_error))),
        "observation_rms": float(np.sqrt(np.mean(observation_error**2))),
        "gait_phase_max": float(
            np.max(np.abs(observation_error[:, 9:11]))
        ),
        "action_max": float(np.max(np.abs(action_error))),
        "action_rms": float(np.sqrt(np.mean(action_error**2))),
        # Touch sensors expose solver contact-force magnitudes. Their active
        # states/transitions are the symmetry invariant; the force split among
        # simultaneous contacts can be numerically non-unique.
        "touch_sensor_value_max": float(np.max(np.abs(contact_value_error))),
        "touch_sensor_value_rms": float(
            np.sqrt(np.mean(contact_value_error**2))
        ),
        "touch_sensor_value_max_after_initial_impact": float(
            np.max(np.abs(contact_value_error[2:]))
        ),
        "contact_state_mismatches": int(np.count_nonzero(contact_a != contact_b)),
        "contact_transition_mismatches": int(
            np.count_nonzero(contact_transition_a != contact_transition_b)
        ),
    }
    print("relative-dynamics metrics:", metrics)

    assert metrics["position_max"] < 1e-7
    assert metrics["orientation_angle_max"] < 1e-7
    assert metrics["linear_velocity_max"] < 1e-6
    assert metrics["angular_velocity_max"] < 1e-6
    assert metrics["joint_position_max"] < 1e-7
    assert metrics["joint_velocity_max"] < 5e-6
    assert metrics["observation_max"] < 5e-6
    assert metrics["gait_phase_max"] == 0.0
    assert metrics["action_max"] < 1e-6
    assert metrics["touch_sensor_value_max_after_initial_impact"] < 5e-5
    assert metrics["contact_state_mismatches"] == 0
    assert metrics["contact_transition_mismatches"] == 0
