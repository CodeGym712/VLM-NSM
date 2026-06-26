"""Stage 22-B: PPO actor with exact Stage 20-Z architecture match.

LSTM(1-layer, 110→256) + LayerNorm(256) + Linear(256,7) = Stage 20-Z action head.
"""

import torch, torch.nn as nn
from torch.distributions import Normal


class ExactMatchPPOActor(nn.Module):
    """PPO actor exactly matching Stage 20-Z LSTM + action head architecture.

    LSTM: 1-layer, input=110, hidden=256, batch_first=True, dropout=0
    Head: LayerNorm(256) → Linear(256, 7)
    """

    def __init__(self, feature_dim=110, action_dim=7, hidden_dim=256, init_log_std=-2.0):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=1, batch_first=True)
        # Stage 20-Z head: Sequential(LayerNorm(256), Linear(256, 7))
        self.head_ln = nn.LayerNorm(hidden_dim)
        self.head_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * init_log_std)

        # Separate critic
        self.critic = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x):
        """x: (B, H, 110) or (H, 110)"""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, (h, _) = self.lstm(x)
        return h[-1]  # (B, 256)

    def get_action_mean(self, latent):
        """Exact Stage 20-Z forward: LayerNorm → Linear."""
        return self.head_linear(self.head_ln(latent))

    def forward(self, x, deterministic=False):
        latent = self.encode(x)
        mean = self.get_action_mean(latent)
        std = torch.exp(self.log_std.clamp(-5, 2))
        dist = Normal(mean, std)
        action = mean if deterministic else dist.rsample()
        action = torch.clamp(action, -1.5, 1.5)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return action, log_prob, value

    def load_stage20z_weights(self, checkpoint_path: str, device: torch.device) -> dict:
        """Load Stage 20-Z LSTM + action head weights.

        Maps: rnn.* → lstm.*, head.0.* → head_ln.*, head.1.* → head_linear.*
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)

        mapping = {
            "rnn.weight_ih_l0": "lstm.weight_ih_l0",
            "rnn.weight_hh_l0": "lstm.weight_hh_l0",
            "rnn.bias_ih_l0": "lstm.bias_ih_l0",
            "rnn.bias_hh_l0": "lstm.bias_hh_l0",
            "head.0.weight": "head_ln.weight",
            "head.0.bias": "head_ln.bias",
            "head.1.weight": "head_linear.weight",
            "head.1.bias": "head_linear.bias",
        }

        own = dict(self.named_parameters())
        loaded, missing, unexpected = [], [], []

        for ckpt_key, local_key in mapping.items():
            if ckpt_key in state and local_key in own:
                if own[local_key].shape == state[ckpt_key].shape:
                    own[local_key].data.copy_(state[ckpt_key].data)
                    loaded.append(ckpt_key)
                else:
                    missing.append(f"{ckpt_key}(shape_mismatch:{list(state[ckpt_key].shape)}→{list(own[local_key].shape)})")
            else:
                missing.append(ckpt_key)

        for ckpt_key in state:
            if ckpt_key not in mapping and not ckpt_key.startswith(("gripper_magnitude_head.", "holding_head.")):
                unexpected.append(ckpt_key)

        return {"loaded": loaded, "missing": missing, "unexpected": unexpected}
