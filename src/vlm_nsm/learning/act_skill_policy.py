"""ACT-lite skill policy for open-loop action chunk prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class ACTSkillPolicy(nn.Module):
    """A compact action-chunk policy conditioned on skill/object IDs.

    This is intentionally lighter than full ACT: observation history is
    flattened, categorical skill/object IDs are embedded, and an MLP regresses
    a fixed-length action chunk.
    """

    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: int = 7,
        obs_history_len: int = 10,
        action_chunk_len: int = 16,
        num_skills: int = 8,
        num_objects: int = 16,
        embedding_dim: int = 16,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_history_len = obs_history_len
        self.action_chunk_len = action_chunk_len
        self.num_skills = num_skills
        self.num_objects = num_objects
        self.embedding_dim = embedding_dim
        self.hidden_dims = list(hidden_dims)

        self.skill_embedding = nn.Embedding(num_skills, embedding_dim)
        self.object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.target_embedding = nn.Embedding(num_objects, embedding_dim)

        input_dim = obs_dim * obs_history_len + embedding_dim * 3
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, action_chunk_len * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_history: torch.Tensor, goal_condition: torch.Tensor) -> torch.Tensor:
        """Predict an action chunk.

        Args:
            obs_history: Tensor ``(B, H, obs_dim)``.
            goal_condition: Long tensor ``(B, 3)`` with
                ``[skill_id, object_id, target_object_id]``.

        Returns:
            Tensor ``(B, action_chunk_len, action_dim)``.
        """
        if obs_history.ndim != 3:
            raise ValueError(f"obs_history must be rank 3, got {tuple(obs_history.shape)}")
        if goal_condition.ndim != 2 or goal_condition.shape[-1] < 3:
            raise ValueError(f"goal_condition must be (B, >=3), got {tuple(goal_condition.shape)}")

        goal_condition = goal_condition.long().clamp_min(0)
        skill_id = goal_condition[:, 0].clamp_max(self.num_skills - 1)
        object_id = goal_condition[:, 1].clamp_max(self.num_objects - 1)
        target_id = goal_condition[:, 2].clamp_max(self.num_objects - 1)

        obs_flat = obs_history.reshape(obs_history.shape[0], -1)
        cond = torch.cat(
            [
                self.skill_embedding(skill_id),
                self.object_embedding(object_id),
                self.target_embedding(target_id),
            ],
            dim=-1,
        )
        out = self.net(torch.cat([obs_flat, cond], dim=-1))
        return out.reshape(obs_history.shape[0], self.action_chunk_len, self.action_dim)

    def config(self) -> dict:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_history_len": self.obs_history_len,
            "action_chunk_len": self.action_chunk_len,
            "num_skills": self.num_skills,
            "num_objects": self.num_objects,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.hidden_dims,
        }
