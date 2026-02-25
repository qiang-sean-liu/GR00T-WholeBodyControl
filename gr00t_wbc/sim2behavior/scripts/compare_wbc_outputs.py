#!/usr/bin/env python3
"""
Parity comparison (step 0 only): Mujoco rollout dump vs OmniGibson wbc_output_compare.

Compares exactly four things at step 0:
  1. Entire inputs to wbc_policy
     Mujoco: observation_sim + wbc_goal
     OmniGibson: wbc_input (observation + goal)
  2. Raw outputs from wbc_policy
     Mujoco: wbc_action (43-D full q)
     OmniGibson: wbc_output (43-D full q)
  3. Entire inputs sent to the simulator/controllers
     Mujoco: sent_to_simulator.robot (or wbc_action)
     OmniGibson: action_sent_to_env + action_sent_to_env_pinocchio_order
  4. Outputs from the simulators after adaption to gr00t input layout
     Mujoco: rollout[1].observation_sim (next-step obs)
     OmniGibson: adapter_output (q, dq, floating_base_pose, floating_base_vel)

Usage:
  python scripts/compare_wbc_outputs.py \\
    output_frames/action_dumps/actions_dump_from_rollout.jsonl \\
    output_frames/action_dumps/wbc_output_compare.jsonl
"""
import json
import math
import sys


def to_float_list(arr):
    if arr is None:
        return []
    if isinstance(arr, dict):
        return []
    return [float(x) for x in arr]


def mj_fb_pose_to_xyzw(pose):
    """Convert Mujoco floating_base_pose [x,y,z, qw,qx,qy,qz] to xyzw [x,y,z, qx,qy,qz,qw].

    Mujoco stores quaternions in wxyz order; OG adapter uses xyzw.
    """
    fl = to_float_list(pose)
    if len(fl) != 7:
        return fl
    x, y, z, qw, qx, qy, qz = fl
    return [x, y, z, qx, qy, qz, qw]


def l2(a, b):
    """L2 norm of (a - b), zero-padding the shorter one."""
    a, b = to_float_list(a), to_float_list(b)
    n = max(len(a), len(b))
    a += [0.0] * (n - len(a))
    b += [0.0] * (n - len(b))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def max_abs_diff(a, b):
    a, b = to_float_list(a), to_float_list(b)
    n = max(len(a), len(b))
    a += [0.0] * (n - len(a))
    b += [0.0] * (n - len(b))
    return max(abs(x - y) for x, y in zip(a, b)) if n else 0.0


def print_vec_compare(name, mj_vec, og_vec, max_elements=10):
    """Print L2 / max-abs diff and first few element-level diffs."""
    mj = to_float_list(mj_vec)
    og = to_float_list(og_vec)
    if not mj and not og:
        print(f"  {name}: (both empty)")
        return
    if not mj:
        print(f"  {name}: Mujoco MISSING, OG has {len(og)} values")
        return
    if not og:
        print(f"  {name}: OG MISSING, Mujoco has {len(mj)} values")
        return
    n = max(len(mj), len(og))
    mj_pad = mj + [0.0] * (n - len(mj))
    og_pad = og + [0.0] * (n - len(og))
    diff = [a - b for a, b in zip(mj_pad, og_pad)]
    l2_val = math.sqrt(sum(d * d for d in diff))
    mad = max(abs(d) for d in diff)
    print(f"  {name} (Mj {len(mj)}-D, OG {len(og)}-D):")
    print(f"    L2 diff = {l2_val:.8f}   max abs diff = {mad:.8f}")
    show = min(max_elements, n)
    for i in range(show):
        print(f"    [{i:2d}] Mj={mj_pad[i]:14.8f}  OG={og_pad[i]:14.8f}  diff={diff[i]:+.4e}")
    if n > show:
        print(f"    ... ({n - show} more elements)")


def ensure_dict(v):
    if v is None:
        return {}
    if isinstance(v, list):
        return v[0] if len(v) > 0 and isinstance(v[0], dict) else {}
    return v if isinstance(v, dict) else {}


def get(rec, key, alt_key=None):
    if rec is None:
        return None
    v = rec.get(key)
    if v is not None:
        return v
    if alt_key:
        return rec.get(alt_key)
    return None


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_wbc_outputs.py <rollout_dump.jsonl> <wbc_compare.jsonl>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        rollout = [json.loads(line) for line in f if line.strip()]
    with open(sys.argv[2]) as f:
        compare = [json.loads(line) for line in f if line.strip()]

    if not rollout or not compare:
        print("No data to compare.")
        return

    mj = rollout[0]
    og = compare[0]

    print(f"Comparing STEP 0 only  (rollout has {len(rollout)} steps, compare has {len(compare)} steps)")
    print()

    # ================================================================
    # 1. Entire inputs to wbc_policy
    # ================================================================
    print("=" * 70)
    print("1. Entire inputs to wbc_policy")
    print("   Mujoco: observation_sim + wbc_goal")
    print("   OmniGibson: wbc_input (observation + goal)")
    print("=" * 70)

    mj_obs = ensure_dict(mj.get("observation_sim"))
    mj_goal = ensure_dict(mj.get("wbc_goal"))
    og_input = ensure_dict(get(og, "wbc_input"))
    og_obs = ensure_dict(og_input.get("observation"))
    og_goal = ensure_dict(og_input.get("goal"))

    print_vec_compare("observation.q", mj_obs.get("q"), og_obs.get("q"))
    print_vec_compare("observation.dq", mj_obs.get("dq"), og_obs.get("dq"))
    mj_fb_pose_input = mj_fb_pose_to_xyzw(mj_obs.get("floating_base_pose"))
    print("  NOTE: Mujoco floating_base_pose converted from wxyz to xyzw for comparison")
    print_vec_compare("observation.floating_base_pose",
                      mj_fb_pose_input, og_obs.get("floating_base_pose"))
    print_vec_compare("observation.floating_base_vel",
                      mj_obs.get("floating_base_vel"), og_obs.get("floating_base_vel"))

    mj_ub = to_float_list(mj_goal.get("target_upper_body_pose"))
    og_ub = to_float_list(og_goal.get("target_upper_body_pose"))
    if len(mj_ub) == 31 and len(og_ub) == 28:
        print(f"  NOTE: Mujoco target_upper_body_pose is 31-D (waist+arms+hands), OG is 28-D (arms+hands).")
        print(f"        Comparing Mj[3:31] vs OG[0:28].")
        mj_ub = mj_ub[3:31]

    print_vec_compare("goal.navigate_cmd",
                      mj_goal.get("navigate_cmd"), og_goal.get("navigate_cmd"))
    print_vec_compare("goal.base_height_command",
                      mj_goal.get("base_height_command"), og_goal.get("base_height_command"))
    print_vec_compare("goal.target_upper_body_pose", mj_ub, og_ub)
    print()

    # ================================================================
    # 2. Raw outputs from wbc_policy
    # ================================================================
    print("=" * 70)
    print("2. Raw outputs from wbc_policy")
    print("   Mujoco: wbc_action  |  OmniGibson: wbc_output")
    print("=" * 70)

    mj_wbc = to_float_list(mj.get("wbc_action"))
    og_wbc = to_float_list(get(og, "wbc_output", "wbc_q"))

    if mj_wbc and og_wbc:
        LOWER = 15
        print_vec_compare("full q (43-D)", mj_wbc, og_wbc, max_elements=15)
        print()
        print_vec_compare("lower body (first 15)",
                          mj_wbc[:LOWER], og_wbc[:LOWER], max_elements=15)
        print_vec_compare("upper body (indices 15:43)",
                          mj_wbc[LOWER:], og_wbc[LOWER:], max_elements=10)
    else:
        print("  Missing wbc_action or wbc_output.")
    print()

    # ================================================================
    # 3. Entire inputs sent to the simulator/controllers
    # ================================================================
    print("=" * 70)
    print("3. Entire inputs sent to the simulator/controllers")
    print("   Mujoco: sent_to_simulator.robot (or wbc_action)")
    print("   OmniGibson: action_sent_to_env + action_sent_to_env_pinocchio_order")
    print("=" * 70)

    mj_sent = ensure_dict(mj.get("sent_to_simulator"))
    mj_sent_vec = to_float_list(mj_sent.get("robot")) or mj_wbc

    og_sent_raw = ensure_dict(get(og, "action_sent_to_env"))
    og_sent_pin = to_float_list(get(og, "action_sent_to_env_pinocchio_order"))

    # OG action_sent_to_env may use "unitree_g1" key
    og_sent_raw_vec = []
    if og_sent_raw:
        for k in og_sent_raw:
            v = to_float_list(og_sent_raw[k])
            if v:
                og_sent_raw_vec = v
                print(f"  OG action_sent_to_env key: '{k}' ({len(v)}-D)")
                break

    if mj_sent_vec and og_sent_pin:
        print_vec_compare("Mj sent_to_simulator.robot vs OG pinocchio_order",
                          mj_sent_vec, og_sent_pin, max_elements=15)
    elif mj_sent_vec and og_sent_raw_vec:
        print(f"  NOTE: No pinocchio_order in OG. Comparing raw (OG controller order ≠ Mujoco order).")
        print_vec_compare("Mj sent_to_simulator.robot vs OG action_sent_to_env (raw)",
                          mj_sent_vec, og_sent_raw_vec, max_elements=15)
    else:
        print("  Missing sent_to_simulator or action_sent_to_env.")
    print()

    # ================================================================
    # 4. Outputs from the simulators after adaption to gr00t input layout
    # ================================================================
    print("=" * 70)
    print("4. Outputs from the simulators after adaption to gr00t input layout")
    print("   Mujoco: rollout[step=1].observation_sim (next-step obs)")
    print("   OmniGibson: adapter_output (q, dq, floating_base_pose, floating_base_vel)")
    print("=" * 70)

    og_adapt = ensure_dict(get(og, "adapter_output", "observation_after_step"))
    og_raw_out = ensure_dict(get(og, "env_step_output_raw"))

    if len(rollout) < 2:
        print("  Mujoco rollout has only 1 step — no next-step observation to compare.")
        if og_adapt:
            print("  OG adapter_output available but no Mujoco counterpart.")
        print()
        return

    mj_next = ensure_dict(rollout[1].get("observation_sim"))

    if not og_adapt:
        print("  No adapter_output in compare file.")
        print()
        return

    print_vec_compare("q", mj_next.get("q"), og_adapt.get("q"), max_elements=15)
    print_vec_compare("dq", mj_next.get("dq"), og_adapt.get("dq"), max_elements=15)
    mj_fb_pose_next = mj_fb_pose_to_xyzw(mj_next.get("floating_base_pose"))
    print("  NOTE: Mujoco floating_base_pose converted from wxyz to xyzw for comparison")
    print_vec_compare("floating_base_pose",
                      mj_fb_pose_next, og_adapt.get("floating_base_pose"),
                      max_elements=7)
    print_vec_compare("floating_base_vel",
                      mj_next.get("floating_base_vel"), og_adapt.get("floating_base_vel"),
                      max_elements=6)

    if og_raw_out:
        print()
        print("  OG raw env output (before adapter):")
        for k in ("joint_positions_og_order", "joint_velocities_og_order", "robot_pos", "robot_quat"):
            v = og_raw_out.get(k)
            if v is not None:
                fv = to_float_list(v)
                print(f"    {k}: {len(fv)}-D")
    print()


if __name__ == "__main__":
    main()
