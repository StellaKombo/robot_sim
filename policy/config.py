from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import numpy as np


@dataclass
class G1Config:

    # PD gains per joint
    Kp: np.ndarray = field(default_factory=lambda: np.array([
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179, 99.098, 40.179, 99.098, 28.501, 28.501,
        40.179,
        14.251, 14.251, 14.251, 14.251, 14.251,
        14.251, 14.251, 14.251, 14.251, 14.251,
    ]))
    Kd: np.ndarray = field(default_factory=lambda: np.array([
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558, 6.309, 2.558, 6.309, 1.814, 1.814,
        2.558,
        0.907, 0.907, 0.907, 0.907, 0.907,
        0.907, 0.907, 0.907, 0.907, 0.907,
    ]))

    # default positions
    default_base_pos: np.ndarray = field(default_factory=lambda: np.array([
        0, 0, 0.78, 1, 0, 0, 0,
    ]))
    default_joint_pos: np.ndarray = field(default_factory=lambda: np.array([
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
         0.0,
         0.35,  0.18, 0.0, 0.87, 0.0,
         0.35, -0.18, 0.0, 0.87, 0.0,
    ]))

    # action scale
    action_scale: np.ndarray = field(default_factory=lambda: np.array([
        0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
        0.548, 0.351, 0.548, 0.351, 0.439, 0.439,
        0.548,
        0.439, 0.439, 0.439, 0.439, 0.439,
        0.439, 0.439, 0.439, 0.439, 0.439,
    ]))

    # control
    control_dt: float = 0.02

    # gait
    gait_period: float = 0.6
    stand_cmd_threshold: float = 0.05 # set to 0.0 to always moving feet

    # velocity command [vx, vy, omega]
    cmd_scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.75]))
    cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


JOG_POLICY_PATH = Path(__file__).resolve().parent / 'jog' / 'g1_29dof_jog_lstm_v0_cont.onnx'


@lru_cache(maxsize=1)
def jog_metadata():
    import onnx
    from policy.jog.jog_actor import JogActor, parse_meta
    model = onnx.load(str(JOG_POLICY_PATH))
    metadata = {p.key: parse_meta(p.value) for p in model.metadata_props}
    metadata['motion_period_frames'] = JogActor._period(metadata)
    if len(metadata['joint_names']) != 29:
        raise ValueError('The selected jogging policy must have 29 ordered joints')
    return metadata


def _jog_array(key):
    value = np.asarray(jog_metadata()[key], dtype=np.float64)
    if value.shape != (29,) or not np.isfinite(value).all():
        raise ValueError(f'Invalid jogging policy metadata: {key}')
    return value.copy()


@dataclass
class JogConfig(G1Config):
    """Deployment parameters from the exact ONNX actor used for KCF collection."""
    Kp: np.ndarray = field(default_factory=lambda: _jog_array('joint_stiffness'))
    Kd: np.ndarray = field(default_factory=lambda: _jog_array('joint_damping'))
    default_joint_pos: np.ndarray = field(default_factory=lambda: _jog_array('default_joint_pos'))
    action_scale: np.ndarray = field(default_factory=lambda: _jog_array('action_scale'))
    default_base_pos: np.ndarray = field(default_factory=lambda: np.array([
        0., 0., .776, 1., 0., 0., 0.,
    ]))
    gait_period: float = field(default_factory=lambda: .02*jog_metadata()['motion_period_frames'])
    stand_cmd_threshold: float = 0.0
    cmd_scale: np.ndarray = field(default_factory=lambda: np.array([1.5, 0., 2.]))


def get_robot_config(state_dim=359):
    if state_dim == 359:
        return JogConfig()
    if state_dim == 85:
        return G1Config()
    raise ValueError('Unsupported robot/model state layout')
