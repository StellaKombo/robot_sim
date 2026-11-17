"""Exercise policy timing and memory with the actual ONNX actor and CPU MuJoCo."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from policy.jog.jog_actor import JogActor
from sim.single import simulation_jog as single
from utils.jog_logging import JogPolicyLog


def check_policy_records(data, policy):
    count = data["policy_time_log"].shape[1]
    assert count > 1
    np.testing.assert_allclose(np.diff(data["policy_time_log"], axis=1), .02, atol=1e-12)
    np.testing.assert_array_equal(data["policy_prev_action_log"][:, 1:], data["policy_action_log"][:, :-1])
    np.testing.assert_array_equal(data["policy_phase_step_log"][:, 1:],
                                  (data["policy_phase_step_log"][:, :-1] + 1) % policy.meta["motion_period_frames"])
    np.testing.assert_allclose(np.linalg.norm(data["policy_clock_log"], axis=-1), 1., atol=1e-6)
    obs = data["policy_obs_log"]
    np.testing.assert_array_equal(obs[..., 61:90], data["policy_prev_action_log"])
    np.testing.assert_array_equal(obs[..., 90:93], data["policy_cmd_log"])
    np.testing.assert_array_equal(obs[..., 93:95], data["policy_gait_phase_log"])
    moving = np.any(data["policy_cmd_log"] != 0., axis=-1)
    np.testing.assert_array_equal(data["policy_gait_phase_log"][~moving], 0.)
    np.testing.assert_allclose(data["policy_gait_phase_log"][moving],
                               data["policy_clock_log"][moving], atol=1e-6)
    np.testing.assert_array_equal(data["terminal_prev_action"], data["policy_action_log"][:, -1])
    np.testing.assert_array_equal(data["terminal_phase_step"],
                                  (data["policy_phase_step_log"][:, -1] + 1) % policy.meta["motion_period_frames"])
    for name in policy.state:
        out = name[:-3] + "_out"
        np.testing.assert_array_equal(data[f"policy_{out}_log"][:, :-1], data[f"policy_{name}_log"][:, 1:])
        np.testing.assert_array_equal(data[f"terminal_{name}"], data[f"policy_{out}_log"][:, -1])
        np.testing.assert_array_equal(data[f"policy_{name}_log"][:, 0], 0.)
    # Replay each query from its saved recurrent inputs. This checks the saved
    # batch/layer ordering and confirms these are the actual inference inputs.
    for k in range(count):
        policy.state = {name: np.moveaxis(data[f"policy_{name}_log"][:, k], 0, 1).copy()
                        for name in policy.state}
        replay = policy(torch.from_numpy(obs[:, k].copy())).numpy()
        np.testing.assert_array_equal(replay, data["policy_action_log"][:, k])
        for name, value in policy.state.items():
            np.testing.assert_array_equal(np.moveaxis(value, 1, 0), data[f"policy_{name[:-3]}_out_log"][:, k])


def drive(sim, commands):
    for command in commands:
        sim.set_cmd(np.asarray(command, dtype=np.float32))
        sim.policy_step()
        for _ in range(sim.decimation):
            sim.step()


@pytest.mark.parametrize("log_hz", [25, 50, 100, 250, 500])
def test_single_saved_policy_records_are_independent_of_physics_log_rate(tmp_path, monkeypatch, log_hz):
    # Only redirect the save destination; model paths remain this checkout's.
    monkeypatch.setitem(single.SimulationBase.save_log.__globals__, "ROOT_DIR", str(tmp_path))
    sim = single.JogSimulation(single.XML_PATH, single.POLICY_PATH, .002,
                               headless=True, raw_cmd=True, log_name="trial", log_hz=log_hz)
    commands = ([[0, 0, 0]] * 4 + [[.8, 0, .1]] * 12 + [[0, 0, 0]] * 7
                + [[-.7, 0, -.2]] * 8 + [[0, 0, 1.2]] * 16)
    drive(sim, commands)
    sim.save_log()
    directory = tmp_path / "data" / "data" / "trial"
    with np.load(directory / sim.full_log_name, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assert data["policy_q_log"].shape == (1, len(commands), 36)
    assert data["policy_h_in_log"].shape == (1, len(commands), 1, 128)
    assert data["policy_log_schema"].item() == "g1_jog_policy_v1"
    assert data["command_source"].item() == "raw_twist"
    assert len(data["policy_sha256"].item()) == 64
    assert data["motion_period_frames"].item() == 43
    np.testing.assert_array_equal(data["policy_mode_log"][0], [0]*4 + [1]*12 + [0]*7 + [2]*8 + [3]*16)
    assert data["terminal_control_aligned"].all()
    np.testing.assert_allclose(data["terminal_time"], len(commands) * .02, atol=1e-12)
    np.testing.assert_allclose(np.diff(data["time_log"], axis=1), 1. / log_hz, atol=1e-12)
    # At the second stop the actor retains its recurrent memory and clock.
    assert np.any(data["policy_h_in_log"][:, 16] != 0.)
    assert data["policy_phase_step_log"][0, 16] == 16
    check_policy_records(data, sim.policy)
    with np.load(directory / sim.rom_log_name, allow_pickle=False) as rom:
        assert "q_rom_log" in rom and "cmd_log" in rom
        assert "policy_h_in_log" not in rom
    sim.reset()
    assert not sim.policy_log.records
    assert not sim.policy_log.terminal
    assert sim.phase_step == 0
    assert all(not value.any() for value in sim.policy.state.values())


def test_single_partial_interval_and_stick_mapping_are_explicit():
    sim = single.JogSimulation(single.XML_PATH, single.POLICY_PATH, .002,
                               headless=True, log_name="partial")
    sim.set_cmd(np.array([.15 * sim.ctrl_scale[0], 0, 0], dtype=np.float32))
    sim.policy_step()
    for _ in range(3):
        sim.step()
    data = sim.full_log_extras()
    assert not data["terminal_control_aligned"].any()
    np.testing.assert_allclose(data["terminal_time"], .006)
    np.testing.assert_array_equal(data["policy_cmd_log"][0, 0], [.5, 0., 0.])
    assert data["command_source"].item() == "shaped_stick"
    np.testing.assert_array_equal(data["terminal_q"][0], sim.data.qpos)


def test_logging_does_not_change_single_policy_or_physics():
    logged = single.JogSimulation(single.XML_PATH, single.POLICY_PATH, .002,
                                  headless=True, raw_cmd=True, log_name="enabled")
    unlogged = single.JogSimulation(single.XML_PATH, single.POLICY_PATH, .002,
                                    headless=True, raw_cmd=True)
    commands = [[0, 0, 0], [.8, 0, -.1], [0, 0, 0], [0, 0, 1.2]]
    drive(logged, commands)
    drive(unlogged, commands)
    np.testing.assert_array_equal(logged.data.qpos, unlogged.data.qpos)
    np.testing.assert_array_equal(logged.data.qvel, unlogged.data.qvel)
    np.testing.assert_array_equal(logged.prev_action, unlogged.prev_action)
    for name in logged.policy.state:
        np.testing.assert_array_equal(logged.policy.state[name], unlogged.policy.state[name])
    assert not unlogged.policy_log.records


def test_saving_after_a_new_query_does_not_mark_its_unexecuted_interval_complete():
    sim = single.JogSimulation(single.XML_PATH, single.POLICY_PATH, .002,
                               headless=True, raw_cmd=True, log_name="pending")
    drive(sim, [[0., 0., 0.]])
    sim.set_cmd(np.array([.8, 0., 0.], dtype=np.float32))
    sim.policy_step()  # Query at 20 ms, interrupted before any subsequent physics.
    data = sim.full_log_extras()
    assert sim.n_steps % sim.decimation == 0
    assert not data["terminal_control_aligned"].any()
    np.testing.assert_array_equal(data["terminal_time"], data["policy_time_log"][:, -1])


def cpu_parallel(monkeypatch, log_hz, batch_size=2):
    """Run the production batched rollout loop with CPU MuJoCo as its stepper.

    Only the unavailable GPU transport/stepper is replaced. Observation, actor,
    command, PD and logging code are the production parallel implementation.
    """
    from sim.parallel import parallel_sim_jog as parallel
    sim = parallel.ParallelJogSim.__new__(parallel.ParallelJogSim)
    sim.cfg = parallel.JogSimConfig(batch_size=batch_size, seed=7, xml_path=single.XML_PATH,
                                   policy_path=single.POLICY_PATH, log_hz=log_hz,
                                   stand_s=.02, cmd_zoh_steps=2)
    sim.device, sim.log_dtype = torch.device("cpu"), torch.float32
    sim.batch_size, sim.nu = batch_size, 29
    sim.rng = torch.Generator().manual_seed(7)
    sim.policy = JogActor(single.POLICY_PATH)
    sim.policy_name = Path(single.POLICY_PATH).name
    meta = sim.policy.meta
    sim.period = meta["motion_period_frames"]
    for name, key in [("default_joint_pos", "default_joint_pos"), ("action_scale", "action_scale"),
                      ("Kp", "joint_stiffness"), ("Kd", "joint_damping")]:
        setattr(sim, name, torch.tensor(meta[key], dtype=torch.float32))
    sim.default_q = np.r_[[0., 0., .776, 1., 0., 0., 0.], meta["default_joint_pos"]].astype(np.float32)
    model = mujoco.MjModel.from_xml_path(single.XML_PATH)
    model.opt.timestep = .002
    sim.nq, sim.nv = model.nq, model.nv
    sim.gyro_adr = int(model.sensor("pelvis_imu_gyro_sensor").adr[0])
    sim.quat_adr = int(model.sensor("pelvis_imu_quat_sensor").adr[0])
    sim.sim_dt, sim.control_dt, sim.decimation, sim.stand_queries = .002, .02, 10, 1
    sim.log_decimation = int(round(1. / log_hz / sim.sim_dt))
    sim.log_dt = sim.log_decimation * sim.sim_dt
    sim.t_qpos = torch.zeros(batch_size, sim.nq)
    sim.t_qvel = torch.zeros(batch_size, sim.nv)
    sim.t_ctrl = torch.zeros(batch_size, sim.nu)
    sim.t_sensordata = torch.zeros(batch_size, model.nsensordata)
    sim.wp_model = model
    sim.wp_data = SimpleNamespace(datas=[mujoco.MjData(model) for _ in range(batch_size)])

    def sync_out(b, data):
        sim.t_qpos[b] = torch.from_numpy(data.qpos.copy())
        sim.t_qvel[b] = torch.from_numpy(data.qvel.copy())
        sim.t_sensordata[b] = torch.from_numpy(data.sensordata.copy())

    def reset(model, state):
        for data in state.datas:
            mujoco.mj_resetData(model, data)

    def forward(model, state):
        for b, data in enumerate(state.datas):
            data.qpos[:] = sim.t_qpos[b].numpy()
            data.qvel[:] = sim.t_qvel[b].numpy()
            mujoco.mj_forward(model, data)
            sync_out(b, data)

    def step(model, state):
        for b, data in enumerate(state.datas):
            data.ctrl[:] = sim.t_ctrl[b].numpy()
            mujoco.mj_step(model, data)
            sync_out(b, data)

    calls = []

    def sample():
        mode, command = [(1, [.8, 0., .1]), (0, [0., 0., 0.]), (3, [0., 0., 1.2])][len(calls) % 3]
        calls.append(mode)
        sim.cmd = torch.tensor(command).repeat(batch_size, 1)
        sim.mode = torch.full((batch_size,), mode, dtype=torch.long)

    monkeypatch.setattr(parallel.wp, "ScopedDevice", lambda _: nullcontext())
    monkeypatch.setattr(parallel.mjwarp, "reset_data", reset)
    monkeypatch.setattr(parallel.mjwarp, "forward", forward)
    monkeypatch.setattr(parallel.mjwarp, "step", step)
    sim._sample_cmd = sample
    return sim


@pytest.mark.parametrize("log_hz", [25, 100, 250])
def test_parallel_control_records_retain_random_phase_and_memory(monkeypatch, tmp_path, log_hz):
    sim = cpu_parallel(monkeypatch, log_hz)
    samples = int(round(.16 / sim.log_dt)) + 1
    q, v, command, mode, fell, fall_time = sim.rollout(samples)
    data = sim.full_log_extras()
    assert data["policy_q_log"].shape == (2, 8, 36)
    expected_phase = torch.randint(0, sim.period, (2,), generator=torch.Generator().manual_seed(7)).numpy()
    np.testing.assert_array_equal(data["policy_phase_step_log"][:, 0], expected_phase)
    assert np.any(expected_phase != 0)
    assert data["terminal_control_aligned"].all()
    np.testing.assert_array_equal(data["terminal_q"], q[:, -1].numpy())
    np.testing.assert_array_equal(data["terminal_v"], v[:, -1].numpy())
    check_policy_records(data, sim.policy)
    # NPZ serialization must work without object arrays or pickle.
    path = tmp_path / "batched.npz"
    np.savez(path, q_log=q.numpy(), v_log=v.numpy(), cmd_log=command.numpy(), **data)
    with np.load(path, allow_pickle=False) as archive:
        for key, value in data.items():
            np.testing.assert_array_equal(archive[key], value)


def test_parallel_partial_interval_and_terminal_only_batch(monkeypatch):
    sim = cpu_parallel(monkeypatch, 500)
    sim.rollout(14)  # 26 ms: one complete transition and one partial interval.
    data = sim.policy_log.arrays()
    assert data["policy_time_log"].shape == (2, 2)
    assert not data["terminal_control_aligned"].any()
    np.testing.assert_allclose(data["terminal_time"], .026)
    sim.rollout(1)
    data = sim.policy_log.arrays()
    assert "policy_time_log" not in data
    np.testing.assert_array_equal(data["terminal_time"], 0.)
    np.testing.assert_array_equal(data["terminal_h_in"], 0.)


def test_mlp_logger_needs_no_recurrent_arrays():
    log = JogPolicyLog(1, 43, .02)
    log.record(time=0., q=np.zeros((1, 36)), v=np.zeros((1, 35)), previous_action=np.zeros((1, 29)),
               command=np.zeros((1, 3)), mode=0, phase_step=8, observation=np.zeros((1, 98)),
               state_in={}, action=np.zeros((1, 29)), state_out={})
    assert log.arrays()["policy_obs_log"].shape == (1, 1, 98)
    assert "policy_h_in_log" not in log.arrays()
