"""Load the frozen 29-DoF walking policy used to train epoch 03240."""

from deploy_robot.utils.policy import Policy


def load_policy(checkpoint_path: str) -> Policy:
    return Policy(checkpoint_path)
