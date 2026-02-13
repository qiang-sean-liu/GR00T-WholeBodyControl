# Review: omnigibson_gr00t_adapters.py

## 1. Observation adaptation vs GR00T policy input

### GR00T (unitree_g1) expected observation

From `gr00t/configs/data/embodiment_configs.py` and `Gr00tSimPolicyWrapper` in `gr00t/policy/gr00t_policy.py`:

- **Video**: `observation["video.<key>"]` with modality_keys `["ego_view"]` → flat key **`"video.ego_view"`**, shape **(B, T, H, W, C)** with T=1, dtype uint8.
- **State**: `observation["state.<key>"]` with modality_keys `["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]` → flat keys **`"state.left_leg"`** etc., shape **(B, T, D)** with T=1, dtype float32.
- **Language**: `observation["annotation.human.task_description"]` (key is the modality key), value **tuple/list of str** (batch dimension B).

The wrapper builds nested `observation["video"]`, `observation["state"]`, `observation["language"]` from these flat keys.

### Adapter output

| Expected (flat) | Adapter provides | Match |
|-----------------|------------------|--------|
| `video.ego_view` (1, 1, H, W, 3) | `obs["video.<mapped_key>"]` where CameraKeyMapper maps `robot0_oak_egoview` → `ego_view` → **`video.ego_view`** | Yes |
| `state.left_leg`, `state.right_leg`, `state.waist`, `state.left_arm`, `state.right_arm`, `state.left_hand`, `state.right_hand` | `obs["state.left_arm"]`, … with `add_batch_time_dims(..., state_time_horizon)` → **(1, T, D)** | Yes |
| `annotation.human.task_description` | `obs["annotation.human.task_description"]` = `("",)` or `(task_description,)` | Yes |

### WBC observation (same adapter output)

WBC policy `G1GearWbcPolicy` uses `observation["q"]`, `observation["dq"]`, `observation["floating_base_pose"]`, `observation["floating_base_vel"]` in **Pinocchio joint order**:

- `observation["q"]`: full joint positions, length **robot_model.num_joints** (43 for G1).
- `observation["dq"]`: same length.
- `observation["floating_base_pose"]`: (7,) [x, y, z, qx, qy, qz, qw].
- `observation["floating_base_vel"]`: (6,) [vx, vy, vz, ωx, ωy, ωz].

The adapter provides all of these; `q`/`dq` are padded/truncated to `robot_model.num_joints`, and base pose/vel come from proprio or run-script extras (`robot_pos`, `robot_quat`, etc.). **Match: yes.**

### Minor notes

- **State order**: Embodiment config state keys are `["left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand"]`. The adapter adds the same keys in the same logical order. OK.
- **Video**: unitree_g1 expects only **ego_view**. The adapter can also add `video.tpp_view` (mid360); the policy only reads `video.ego_view`. OK.
- **State shape**: Adapter uses `state_time_horizon` (default 1) → (1, 1, D). Policy expects T=1. OK.

**Verdict: Observation adaptation matches GR00T (and Gr00tSimPolicyWrapper) and WBC input.**

---

## 2. Action adaptation vs WBC output

### WBC output

- **G1DecoupledWholeBodyPolicy** returns `{"q": q}` where **q** is **43-D in Pinocchio (robot_model) joint order**: q[0:15] = lower body (legs + waist), q[15:43] = upper body.

### Adapter behavior for `"q"`

- When `gr00t_action["q"]` is present, the adapter uses **`_map_joints_to_og_action_space(q, action_space)`**:
  - If `len(q) == self.og_action_dim` (e.g. 43 == 29 is false), it does **not** pass through unchanged.
  - It builds a name-based mapping from robot_model joint names to `og_joint_names` or falls back to body indices / first `og_action_dim` indices.
  - So the adapter returns a vector of length **og_action_dim** (e.g. 29), but that vector can be **wrong** if it was derived by indexing Pinocchio-ordered `q` with articulation/controller indices.

### Run script (correct behavior for WBC)

In **run_omnigibson_g1_with_gr00t.py**:

- When **full WBC** `q` is present (`gr00t_action["q"]` length 43), the script **does not** use the adapter’s robot action for the final step.
- It sets  
  `og_action[robot_name_in_config] = _to_robot_action_dim(robot, _pinocchio_q_to_controller_order_action(robot, gr00t_action["q"], robot_model))`  
  i.e. it converts **Pinocchio `q` → controller order by joint name**, then trims/pads to **robot.action_dim** (e.g. 29).

So for the **WBC path**, the **action adapter’s** `_map_joints_to_og_action_space` result for the robot is **overwritten**; the final sim action comes from the name-based conversion in the run script. **That design is correct** and avoids using Pinocchio-ordered `q` as if it were in controller/dof order.

### Adapter when used alone (no run-script override)

- If you called **only** `action_adapter.adapt(gr00t_action)` with WBC’s 43-D `q` and no run-script logic:
  - With `og_action_dim == 29`, the adapter would **not** return `q` unchanged (43 ≠ 29).
  - It would use mapping or `q[:29]` / body_indices; **ordering would be wrong** for OmniGibson (Pinocchio vs controller order).
- So the adapter’s **robot** output is **only correct** when either:
  - The run script overwrites it (WBC path), or
  - The action is already in OmniGibson/controller order and dimension (e.g. from another source).

**Verdict: Action adaptation does not by itself match WBC output to the sim (joint order/dim differ). The run script correctly compensates by replacing the robot action with name-based Pinocchio→controller conversion and length enforcement. For the full pipeline (run script + adapter), action handling matches WBC output.**

---

## 3. Summary

| Component | Match |
|-----------|--------|
| Observation → GR00T (flat keys, shapes, video/state/language) | Yes |
| Observation → WBC (q, dq, floating_base_pose/vel) | Yes |
| Action: adapter output for robot when using WBC 43-D q | Overwritten by run script (intended) |
| Action: run script final robot action from WBC q | Yes (name-based, controller order, action_dim) |

No code changes are required in the adapters for the current run script and WBC pipeline. The run script’s use of `_pinocchio_q_to_controller_order_action` and `_to_robot_action_dim` is what makes action adaptation match WBC output for the sim.
