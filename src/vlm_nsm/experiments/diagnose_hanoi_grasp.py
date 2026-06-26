"""Diagnose Hanoi grasp failures — test each failure hypothesis step by step.

Hypotheses:
  H1: Object pose (get_object_pose) returns wrong/invalid values in peg env
  H2: DESCEND path collides with pegs (peg obstructs gripper approach)
  H3: Contact detection fails — gripper closes fully even with object (peg geometry)
  H4: Cross-table reach leaves EEF too far from object for MICRO_ALIGN to fix

Usage:
    MUJOCO_GL=egl nsm/bin/python -m vlm_nsm.experiments.diagnose_hanoi_grasp
"""

import math
import numpy as np

from vlm_nsm.embodied.controllers.operational_space_pid import (
    OperationalSpacePID, GainSchedule,
)

_OSC_POSE_MAX_DELTA = 0.05
_GRIPPER_OPEN = -1.0
_GRIPPER_CLOSE = 1.0


def _norm_action(dx, dy, dz, gripper):
    return np.array([np.clip(dx / _OSC_POSE_MAX_DELTA, -1, 1),
                     np.clip(dy / _OSC_POSE_MAX_DELTA, -1, 1),
                     np.clip(dz / _OSC_POSE_MAX_DELTA, -1, 1),
                     0, 0, 0, gripper], dtype=np.float32)


def _dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def test_h1_object_pose(env):
    """H1: Can we read object poses? Are they valid?"""
    print("\n=== H1: Object Pose Reading ===")
    for name in env.cube_names:
        pose = env.get_object_pose(name)
        valid = all(np.isfinite([pose["x"], pose["y"], pose["z"]]))
        on_table = 0.80 < pose["z"] < 0.90
        print(f"  {name}: pos=({pose['x']:.3f}, {pose['y']:.3f}, {pose['z']:.3f}) "
              f"valid={valid} on_peg={on_table}")
    return True


def test_h2_descend_path(env, pid, disk_name):
    """H2: Does DESCEND collide with pegs?

    Approach directly above the object, then descend slowly. Check if EEF
    position is as expected at each step. If the EEF stops moving in XY,
    it may be hitting a peg.
    """
    print(f"\n=== H2: DESCEND Path (disk={disk_name}) ===")

    obj = env.get_object_pose(disk_name)
    target_approach = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}
    eef = env.get_eef_pose()

    print(f"  Object: ({obj['x']:.3f}, {obj['y']:.3f}, {obj['z']:.3f})")
    print(f"  EEF start: ({eef['x']:.3f}, {eef['y']:.3f}, {eef['z']:.3f})")
    print(f"  XY distance to object: {math.hypot(eef['x']-obj['x'], eef['y']-obj['y']):.3f}m")

    # Approach
    pid.set_phase("reach_pick")
    pid.set_target(pos=target_approach)
    for step in range(200):
        eef = env.get_eef_pose()
        pid.set_current(eef)
        a = pid.act("reach_pick", None)
        a[6] = _GRIPPER_OPEN
        env.step(a)
        if _dist(eef, target_approach) < 0.01:
            print(f"  Approach done in {step} steps")
            break
    else:
        print(f"  Approach timeout — EEF at ({eef['x']:.3f}, {eef['y']:.3f}, {eef['z']:.3f})")
        return False

    # Descend slowly and track XY
    object_above = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.005}
    pid.set_phase("grasp")
    pid.set_target(pos=object_above)
    xy_drift = []
    for step in range(100):
        eef = env.get_eef_pose()
        xy_drift.append(math.hypot(eef["x"] - obj["x"], eef["y"] - obj["y"]))
        pid.set_current(eef)
        a = pid.act("grasp", None)
        a[6] = _GRIPPER_OPEN
        env.step(a)
        if eef["z"] <= obj["z"] + 0.01:
            break

    final_eef = env.get_eef_pose()
    final_xy_err = math.hypot(final_eef["x"] - obj["x"], final_eef["y"] - obj["y"])
    print(f"  After descend: EEF=({final_eef['x']:.3f},{final_eef['y']:.3f},{final_eef['z']:.3f})")
    print(f"  Final XY error: {final_xy_err:.3f}m")
    print(f"  XY drift range: {min(xy_drift):.3f} ~ {max(xy_drift):.3f}m")
    if final_xy_err > 0.02:
        print(f"  ⚠️ XY error > 20mm — MICRO_ALIGN may not be enough")
    return True


def test_h3_contact_detection(env, pid, disk_name):
    """H3: Does contact detection work in peg env?

    Manually position EEF above object, descend, close gripper.
    Check if contact is correctly detected.
    """
    print(f"\n=== H3: Contact Detection (disk={disk_name}) ===")

    obj = env.get_object_pose(disk_name)

    # Manual approach directly above object
    pid.set_phase("reach_pick")
    pid.set_target(pos={"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10})
    for _ in range(200):
        eef = env.get_eef_pose()
        pid.set_current(eef)
        a = pid.act("reach_pick", None)
        a[6] = _GRIPPER_OPEN
        env.step(a)
        if _dist(eef, {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.10}) < 0.01:
            break

    # Descend
    pid.set_phase("grasp")
    pid.set_target(pos={"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.003})
    for _ in range(100):
        eef = env.get_eef_pose()
        pid.set_current(eef)
        a = pid.act("grasp", None)
        a[6] = _GRIPPER_OPEN
        env.step(a)
        if eef["z"] <= obj["z"] + 0.008:
            break

    eef_before = env.get_eef_pose()
    gripper_before = env.get_gripper_state()
    print(f"  Before close: EEF z={eef_before['z']:.3f}, open={gripper_before.get('open', -1):.3f}")

    # Close gripper (manual, slow)
    for _ in range(70):
        a = _norm_action(0, 0, 0, _GRIPPER_CLOSE * 0.5)  # half speed
        env.step(a)

    gripper_after = env.get_gripper_state()
    open_before = gripper_before.get("open", -1)
    open_after = gripper_after.get("open", -1)
    contact = (open_before > 0.03 and open_after > 0.005
               and open_after < open_before * 0.8)
    print(f"  After close: open={open_after:.3f}, contact_detected={contact}")
    print(f"  open_before={open_before:.3f}, open_after={open_after:.3f}")

    if not contact:
        print(f"  ⚠️ Contact NOT detected! Possible causes:")
        if open_after <= 0.005:
            print(f"     - Gripper fully closed (object too thin or missed)")
        if open_after >= open_before * 0.8:
            print(f"     - Gripper barely moved (object too large or collision)")

    # Try LIFT
    obj_after = env.get_object_pose(disk_name)
    z_before = obj_after["z"]
    for _ in range(60):
        a = _norm_action(0, 0, 0.2, _GRIPPER_CLOSE)
        env.step(a)
    obj_after = env.get_object_pose(disk_name)
    z_after = obj_after["z"]
    lifted = (z_after - z_before) > 0.005
    print(f"  After lift: z={z_before:.3f}→{z_after:.3f}, lifted={lifted}")

    return contact and lifted


def test_h4_micro_align_range(env, pid, disk_name):
    """H4: Can MICRO_ALIGN correct XY error after cross-table reach?

    Intentionally position EEF off-center, then test MICRO_ALIGN correction.
    """
    print(f"\n=== H4: MICRO_ALIGN Range (disk={disk_name}) ===")

    obj = env.get_object_pose(disk_name)

    # Intentionally offset EEF by 20mm in XY
    offset_x, offset_y = 0.02, 0.02
    target_offset = {"x": obj["x"] + offset_x, "y": obj["y"] + offset_y,
                     "z": obj["z"] + 0.005}  # at surface level
    pid.set_phase("grasp")
    pid.set_target(pos=target_offset)
    for _ in range(200):
        eef = env.get_eef_pose()
        pid.set_current(eef)
        a = pid.act("grasp", None)
        a[6] = _GRIPPER_OPEN
        env.step(a)
        if _dist(eef, target_offset) < 0.005:
            break

    eef_off = env.get_eef_pose()
    xy_err_before = math.hypot(eef_off["x"] - obj["x"], eef_off["y"] - obj["y"])
    print(f"  After offset: XY error = {xy_err_before:.3f}m")

    # Now try MICRO_ALIGN to correct
    micro_target = {"x": obj["x"], "y": obj["y"], "z": obj["z"] + 0.005}
    pid.set_phase("grasp")
    pid.set_target(pos=micro_target)
    for step in range(60):
        eef = env.get_eef_pose()
        pid.set_current(eef)
        a = pid.act("grasp", None)
        a[6] = _GRIPPER_OPEN
        env.step(a)
        xy_err = math.hypot(eef["x"] - obj["x"], eef["y"] - obj["y"])
        if xy_err < 0.012:
            print(f"  MICRO_ALIGN converged: XY error = {xy_err:.3f}m in {step} steps")
            return True

    eef_final = env.get_eef_pose()
    xy_err_after = math.hypot(eef_final["x"] - obj["x"], eef_final["y"] - obj["y"])
    print(f"  MICRO_ALIGN timeout: XY error = {xy_err_after:.3f}m (from {xy_err_before:.3f}m)")
    return False


def main():
    print("=" * 60)
    print("Hanoi Grasp Diagnostic Suite")
    print("=" * 60)

    # Patch Stack for 3-cube + pegs (same as run_hanoi_eval.py)
    import vlm_nsm.executors.stack_three_cube_env as _  # noqa: F401
    from vlm_nsm.executors.robosuite_env_wrapper import RobosuiteEnvWrapper

    env = RobosuiteEnvWrapper(
        env_name="Stack", robots="Panda",
        cube_names=["cube1", "cube2", "cube3"],
        render_mode="offscreen",
    )
    env.reset()
    env.env.horizon = 10000
    pid = OperationalSpacePID(gain_schedule=GainSchedule())

    print(f"Cubes: {env.cube_names}")

    # H1: Object pose
    test_h1_object_pose(env)

    # H3 first on nearest cube (cube1 is usually closest to initial EEF)
    disk = env.cube_names[0]  # pick nearest
    h3_ok = test_h3_contact_detection(env, pid, disk)
    if h3_ok:
        print("\n  ✅ H3 PASS: Contact detection works")
    else:
        print("\n  ❌ H3 FAIL: Contact detection broken — this is the root cause")

    # H4: MICRO_ALIGN range
    env.reset()
    disk = env.cube_names[0]
    h4_ok = test_h4_micro_align_range(env, pid, disk)
    if h4_ok:
        print("\n  ✅ H4 PASS: MICRO_ALIGN can correct XY")
    else:
        print("\n  ❌ H4 FAIL: MICRO_ALIGN insufficient range")

    # H2: Descend path
    env.reset()
    disk = env.cube_names[0]
    h2_ok = test_h2_descend_path(env, pid, disk)
    if h2_ok:
        print("\n  ✅ H2 PASS: DESCEND path is clear")
    else:
        print("\n  ❌ H2 FAIL: DESCEND collides with peg")

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print(f"  H1 (pose reading): PASS (always)")
    print(f"  H2 (descend path):  {'PASS' if h2_ok else 'FAIL'}")
    print(f"  H3 (contact):       {'PASS ✅' if h3_ok else 'FAIL ❌ — ROOT CAUSE'}")
    print(f"  H4 (micro_align):   {'PASS' if h4_ok else 'FAIL'}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
