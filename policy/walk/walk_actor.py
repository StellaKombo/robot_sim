##
#
#  Actor MLP and checkpoint loader for the RSL-RL G1 policy.
#
##

import torch
import torch.nn as nn


class ActorMLP(nn.Module):
    """RSL-RL actor MLP with observation normalization."""

    def __init__(self, input_size=80,
                       hidden_sizes=(512, 256, 128),
                       output_size=23):
        super().__init__()

        # build the MLP
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ELU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.mlp = nn.Sequential(*layers)

        # observation normalization buffers
        self.register_buffer("obs_mean", torch.zeros(1, input_size))
        self.register_buffer("obs_var", torch.ones(1, input_size))

    # forward pass with no gradients (inference only)
    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs_norm = (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-2)
        return self.mlp(obs_norm)


# load a pytorch policy
def load_policy(checkpoint_path: str, device: torch.device) -> ActorMLP:

    # load the checkpoint and extract the actor state dict
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_sd = ckpt["actor_state_dict"]

    # load the policy onto device
    policy = ActorMLP().to(device)
    weight_map = {}
    for k, v in actor_sd.items():
        if k.startswith("mlp."):
            weight_map[k] = v
        elif k == "obs_normalizer._mean":
            weight_map["obs_mean"] = v
        elif k == "obs_normalizer._var":
            weight_map["obs_var"] = v

    # load weights into policy
    policy.load_state_dict(weight_map, strict=False)
    policy.eval()

    return policy
