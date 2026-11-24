"""Load the frozen 29-DoF walking policy used to train epoch 03240."""

import onnxruntime as ort
from deploy_robot.utils.policy import Policy


def load_policy(checkpoint_path: str) -> Policy:
    # Keep the simulator's inference pool from competing with the MPC worker.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    policy = Policy(checkpoint_path)
    policy._onnx_session = ort.InferenceSession(
        policy.policy.SerializeToString(), sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return policy
