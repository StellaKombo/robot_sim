##
#
#  Batched onnxruntime actor for the 29-DoF jog policies (MLP or LSTM ONNX exports).
#  The exports are batch-1, so the batch dim is made dynamic in memory (the file is untouched).
#  An LSTM's (h, c) is carried per env across calls; reset() zeroes it.
#
##

import json

import numpy as np
import onnx
import onnxruntime as ort
import torch


# metadata value -> python: JSON, float CSV, string CSV, or plain string
def parse_meta(value):
    if value.lstrip().startswith(("[", "{")):
        try:
            return json.loads(value)
        except ValueError:
            pass
    parts = [p.strip() for p in value.split(",") if p.strip()]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return parts if "," in value else value.strip()


class JogActor:
    """obs (B, 98) torch -> actions (B, 29) torch on the same device, via onnxruntime."""

    def __init__(self, onnx_path: str):
        model = onnx.load(onnx_path)
        self.meta = {p.key: parse_meta(p.value) for p in model.metadata_props}
        self.meta["motion_period_frames"] = self._period(self.meta)

        # obs/actions batch is dim 0; LSTM state (1, batch, H) is dim 1
        for vi in list(model.graph.input) + list(model.graph.output):
            dims = vi.type.tensor_type.shape.dim
            dims[1 if vi.name.startswith(("h_", "c_")) else 0].dim_param = "batch"
        del model.graph.value_info[:]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model.SerializeToString(), opts,
                                            providers=ort.get_available_providers())
        self.obs_name = self.session.get_inputs()[0].name
        self.out_names = [o.name for o in self.session.get_outputs()]
        self.state_shapes = {i.name: i.shape for i in self.session.get_inputs()[1:]}
        self.kind = "lstm" if self.state_shapes else "mlp"
        self.state = {}

    @staticmethod
    def _period(meta):
        if "motion_period_frames" in meta:
            return int(meta["motion_period_frames"][0])
        periods = {int(c[3]) for c in meta["library_clips"]}
        assert len(periods) == 1, f"mixed gait periods {periods}: not a single-stride jog policy"
        return periods.pop()

    # zero the recurrent state for B envs (an episode start, as in training)
    def reset(self, batch_size: int):
        self.state = {n: np.zeros([batch_size if d == "batch" else d for d in s], np.float32)
                      for n, s in self.state_shapes.items()}

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        feed = {self.obs_name: obs.detach().float().cpu().numpy(), **self.state}
        res = dict(zip(self.out_names, self.session.run(None, feed)))
        for n in self.state:
            self.state[n] = res[n[:-3] + "_out"]
        return torch.from_numpy(res[self.out_names[0]]).to(obs.device)
