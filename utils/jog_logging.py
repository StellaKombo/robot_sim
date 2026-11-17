"""Control-rate jogging records, independent of the physics logging frequency."""

import hashlib
import json
from pathlib import Path

import numpy as np


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class JogPolicyLog:
    """Record pre-query robot/actor state and the output of that same query.

    All saved arrays are batch first. ONNX (layers, batch, hidden) memory is
    moved to (batch, layers, hidden), then stacked along the query axis. The
    ungated clock is retained during standing; the masked phase is copied from
    the actual policy observation. No policy replay or extra inference occurs.
    """

    def __init__(self, batch_size, period, control_dt):
        self.batch_size = batch_size
        self.period = period
        self.control_dt = control_dt
        self.records = {}
        self.terminal = {}

    def _batch_scalar(self, value, dtype):
        return np.broadcast_to(_numpy(value), (self.batch_size,)).astype(dtype, copy=True)

    def _clock(self, phase_step):
        angle = 2. * np.pi * phase_step.astype(np.float64) / self.period
        return np.stack((np.sin(angle), np.cos(angle)), axis=-1).astype(np.float32)

    @staticmethod
    def _memory(state):
        return {name: np.moveaxis(_numpy(value), 1, 0).copy()
                for name, value in state.items()}

    def record(self, *, time, q, v, previous_action, command, mode, phase_step,
               observation, state_in, action, state_out):
        phase = self._batch_scalar(phase_step, np.int64)
        obs = _numpy(observation)
        values = dict(
            policy_time_log=self._batch_scalar(time, np.float64),
            policy_q_log=_numpy(q).copy(),
            policy_v_log=_numpy(v).copy(),
            policy_prev_action_log=_numpy(previous_action).copy(),
            policy_cmd_log=_numpy(command).copy(),
            policy_mode_log=self._batch_scalar(mode, np.int8),
            policy_phase_step_log=phase,
            policy_clock_log=self._clock(phase),
            policy_gait_phase_log=obs[..., 93:95].copy(),
            policy_obs_log=obs.copy(),
            policy_action_log=_numpy(action).copy(),
        )
        for name, value in self._memory(state_in).items():
            values[f"policy_{name}_log"] = value
        for name, value in self._memory(state_out).items():
            values[f"policy_{name[:-3]}_out_log"] = value
        if self.records and self.records.keys() != values.keys():
            raise ValueError("Jog policy recurrent inputs changed during the rollout")
        for name, value in values.items():
            self.records.setdefault(name, []).append(value)

    def finish(self, *, time, q, v, previous_action, phase_step, state, control_aligned):
        """Retain the final robot/controller state, including partial intervals."""
        phase = self._batch_scalar(phase_step, np.int64)
        terminal_time = self._batch_scalar(time, np.float64)
        aligned = self._batch_scalar(control_aligned, np.bool_)
        if self.records:
            elapsed = terminal_time - self.records["policy_time_log"][-1]
            aligned &= np.isclose(elapsed, self.control_dt, rtol=0., atol=1e-9)
        else:
            aligned[:] = False
        self.terminal = dict(
            terminal_time=terminal_time,
            terminal_q=_numpy(q).copy(),
            terminal_v=_numpy(v).copy(),
            terminal_prev_action=_numpy(previous_action).copy(),
            terminal_phase_step=phase,
            terminal_clock=self._clock(phase),
            terminal_control_aligned=aligned,
        )
        self.terminal.update({f"terminal_{name}": value
                              for name, value in self._memory(state).items()})

    def arrays(self):
        return {**{name: np.stack(values, axis=1) for name, values in self.records.items()},
                **self.terminal}


def jog_log_metadata(policy, policy_path, model_path, control_dt):
    """Version the alignment convention and identify the actual policy/model."""
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return dict(
        policy_log_schema=np.array("g1_jog_policy_v1"),
        policy_kind=np.array(policy.kind),
        policy_state_names=np.array(list(policy.state_shapes), dtype=str),
        policy_sha256=np.array(digest(policy_path)),
        model_sha256=np.array(digest(model_path)),
        joint_names=np.asarray(policy.meta["joint_names"], dtype=str),
        motion_period_frames=np.int64(policy.meta["motion_period_frames"]),
        gait_period=np.float64(policy.meta["motion_period_frames"] * control_dt),
        policy_log_convention=np.array(json.dumps(dict(
            sample="policy_* rows describe the state immediately before query k",
            output="policy_action_log and *_out_log are outputs of query k",
            transition="command[k] drives state[k] to state[k+1]",
            memory_layout="batch, query, layers, hidden",
            clock="ungated free-running clock; policy_gait_phase_log is the masked input",
            terminal="final state; use the last transition only if terminal_control_aligned",
            physics_commands="cmd_log and mode_log describe the command at the physics sample",
        ), sort_keys=True)),
    )
