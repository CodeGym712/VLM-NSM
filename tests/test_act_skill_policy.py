import pytest
import torch

from vlm_nsm.learning.act_skill_policy import ACTSkillPolicy


def test_act_skill_policy_forward_shape():
    model = ACTSkillPolicy(
        obs_dim=11,
        action_dim=7,
        obs_history_len=10,
        action_chunk_len=16,
        hidden_dims=[32],
        embedding_dim=8,
    )
    obs_history = torch.zeros(4, 10, 11)
    goal_condition = torch.tensor([[0, 2, 3]] * 4)

    out = model(obs_history, goal_condition)

    assert out.shape == (4, 16, 7)


def test_act_skill_policy_config_roundtrip_fields():
    model = ACTSkillPolicy(obs_dim=5, action_dim=2, obs_history_len=3, action_chunk_len=4)
    cfg = model.config()

    assert cfg["obs_dim"] == 5
    assert cfg["action_dim"] == 2
    assert cfg["obs_history_len"] == 3
    assert cfg["action_chunk_len"] == 4


def test_act_skill_policy_rejects_bad_obs_rank():
    model = ACTSkillPolicy()
    with pytest.raises(ValueError, match="obs_history"):
        model(torch.zeros(2, 11), torch.zeros(2, 3, dtype=torch.long))


def test_act_skill_policy_rejects_bad_goal_shape():
    model = ACTSkillPolicy()
    with pytest.raises(ValueError, match="goal_condition"):
        model(torch.zeros(2, 10, 11), torch.zeros(2, dtype=torch.long))
