from vlm_nsm.core.action_grounder import ActionGrounder
from vlm_nsm.core.data_types import Action
from vlm_nsm.executors.mock_env_wrapper import MockEnvWrapper


def test_action_grounder_places_on_cube_with_correct_height():
    env = MockEnvWrapper()
    grounder = ActionGrounder(env, block_height=0.04)
    grounded = grounder.ground_move(Action("move", ["cube1", "table", "cube2"]))
    assert grounded["grasp_pose"]["z"] > grounded["object_pose"]["z"]
    assert grounded["place_pose"]["z"] == grounded["target_pose"]["z"] + 0.04


def test_action_grounder_places_on_table_half_block_height():
    env = MockEnvWrapper()
    grounder = ActionGrounder(env, block_height=0.04)
    grounded = grounder.ground_move(Action("move", ["cube1", "cube2", "table"]))
    assert grounded["place_pose"]["z"] == 0.02
