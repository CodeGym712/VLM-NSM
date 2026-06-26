"""Tests for Behavior Tree — Commit 5."""

from vlm_nsm.embodied.behavior_tree import (
    BTStatus, BTTrace, BTNodeTrace,
    Sequence, Fallback, Condition, Action, Retry, Verify,
    stacking_2block_tree,
)


class TestBTNodes:
    """Test individual node construction — not execution (needs MuJoCo)."""

    def test_sequence_construction(self):
        seq = Sequence(children=[], name="test")
        assert seq.name == "test"
        assert len(seq.children) == 0

    def test_fallback_construction(self):
        fb = Fallback(children=[], name="fb")
        assert fb.name == "fb"

    def test_condition_construction(self):
        c = Condition("clear", "cube1")
        assert c.pred_name == "clear"
        assert c.args == ["cube1"]

    def test_action_construction(self):
        a = Action("reach", target_object="cube1")
        assert a.skill_name == "reach"
        assert a.params["target_object"] == "cube1"

    def test_retry_construction(self):
        a = Action("grasp", target_object="cube1")
        r = Retry(a, max_retries=3, strategy="continue")
        assert r.max_retries == 3
        assert r.strategy == "continue"

    def test_verify_construction(self):
        v = Verify("on", "cube1", "cube2")
        assert v.pred_name == "on"
        assert v.args == ["cube1", "cube2"]

    def test_stacking_tree_structure(self):
        tree = stacking_2block_tree("cube1", "cube2")
        assert isinstance(tree, Sequence)
        assert len(tree.children) == 5  # reach, grasp, lift, reach_place, Fallback
        fallback = tree.children[4]
        assert isinstance(fallback, Fallback)
        assert len(fallback.children) == 2  # PlaceAndVerify, RecoverAndRetry
        # PlaceAndVerify = Sequence(Action, Verify)
        pv = fallback.children[0]
        assert isinstance(pv, Sequence)
        assert isinstance(pv.children[0], Action)  # release
        assert isinstance(pv.children[1], Verify)  # verify
        # RecoverAndRetry = Sequence(Action, Action, Retry(Action), Action, Action, Action)
        rr = fallback.children[1]
        assert isinstance(rr, Sequence)
        assert isinstance(rr.children[2], Retry)  # retry grasp

    def test_tree_node_count(self):
        """Verify the stacking tree has the right number of leaf actions."""
        def count_nodes(node):
            if isinstance(node, (Sequence, Fallback)):
                return 1 + sum(count_nodes(c) for c in node.children)
            elif isinstance(node, Retry):
                return 1 + count_nodes(node.child)
            return 1  # Action, Condition, Verify

        tree = stacking_2block_tree()
        n = count_nodes(tree)
        assert 15 <= n <= 25  # reasonable range for a stacking tree


class TestBTTrace:
    def test_empty_trace(self):
        t = BTTrace()
        assert t.total_steps == 0
        assert len(t.nodes) == 0

    def test_record(self):
        t = BTTrace()
        t.record(BTNodeTrace(node_type="Action", node_name="test",
                             status=BTStatus.SUCCESS, steps=42))
        assert len(t.nodes) == 1
        assert t.total_steps == 42

    def test_summary(self):
        t = BTTrace()
        t.record(BTNodeTrace(node_type="Action", node_name="reach",
                             status=BTStatus.SUCCESS, steps=100))
        t.record(BTNodeTrace(node_type="Action", node_name="grasp",
                             status=BTStatus.FAILURE, steps=50,
                             failure_type="nocontact"))
        t.final_status = BTStatus.FAILURE
        s = t.summary()
        assert "reach" in s
        assert "grasp" in s
        assert "nocontact" in s
        assert "FAIL" in s

    def test_to_dict(self):
        t = BTTrace()
        t.record(BTNodeTrace(node_type="Verify", node_name="check",
                             status=BTStatus.SUCCESS))
        d = t.to_dict()
        assert d["n_nodes"] == 1
        assert d["nodes"][0]["type"] == "Verify"
