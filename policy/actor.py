"""Frozen running actor and metadata-driven variable-stride clock."""
import json
import numpy as np
import onnxruntime as ort
from policy.config import policy_metadata


def load_policy(checkpoint_path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(checkpoint_path), sess_options=options,
                                providers=["CPUExecutionProvider"])


class GaitClock:
    """Per-clip frame clock with deployment's phase-preserving remap."""
    def __init__(self):
        meta = policy_metadata()
        clips = np.asarray(json.loads(meta["library_clips"]), dtype=np.float64)
        self.twists, self.periods = clips[:, :3], clips[:, 3].astype(int)
        self.weights = np.asarray(json.loads(meta["twist_metric_weights"]))
        self.idle = int(np.flatnonzero(np.linalg.norm(self.twists, axis=-1) < 1e-4)[0])
        self.reset()

    def reset(self):
        self.clip, self.frame = self.idle, 0

    @property
    def state(self):
        period = int(self.periods[self.clip])
        angle = 2. * np.pi * self.frame / period
        return np.array([np.sin(angle), np.cos(angle), period], dtype=np.float32)

    def update(self, command):
        clip = int(np.argmin(np.sum((self.twists - command)**2 * self.weights, axis=-1)))
        old, new = self.periods[self.clip], self.periods[clip]
        if old != new:
            self.frame = int(np.round(self.frame / old * new)) % new
        self.clip = clip
        return np.zeros(2, dtype=np.float32) if clip == self.idle else self.state[:2]

    def advance(self):
        self.frame = (self.frame + 1) % self.periods[self.clip]
