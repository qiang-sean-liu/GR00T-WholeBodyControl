# sim2behavior: Design for an OmniGibson/BEHAVIOR Version of sim2mujoco

This document describes how to add a **sim2behavior/** package that mirrors **sim2mujoco/** but uses **BEHAVIOR/OmniGibson** as the simulator instead of MuJoCo. It assumes you already have the [GR00T OmniGibson Deployment Guide](../../../GR00T_OmniGibson_Deployment_Guide.md) and the existing `gr00t/eval/sim/BEHAVIOR` integration (OmniGibson + BEHAVIORGr00tEnv) in place.

---

## 1. What sim2mujoco does (recap)

| Component | Role |
|-----------|------|
| **Robot model** | G1 URDF/MJCF + meshes for MuJoCo |
| **WBC config** | `g1_gear_wbc.yaml`: PD gains, policy paths, obs/action dims |
| **Lower-body policies** | Balance.onnx, Walk.onnx (ONNX) for legs/waist |
| **Standalone script** | `run_mujoco_gear_wbc.py`: pure MuJoCo sim + WBC + keyboard |
| **Integration** | `G1GearWbcPolicy` and teleop load config/ONNX from sim2mujoco |

---

## 2. What sim2behavior should provide (parallel structure)

| sim2mujoco | sim2behavior (target) |
|------------|-------------------------|
| G1 MJCF/URDF + STL meshes | G1 description in **OmniGibson-compatible format** (URDF + meshes, or USD if required by OmniGibson) |
| `g1_gear_wbc.yaml` | **Controller + GR00T interface config** for OmniGibson (YAML/JSON): controller types, joint groups, action mapping, obs keys |
| Balance/Walk ONNX | **Optional**: same ONNX or OmniGibson-side WBC adapter; or rely on OmniGibson’s built-in controllers and only map GR00T actions |
| `run_mujoco_gear_wbc.py` | **Standalone script**: launch OmniGibson env with G1, optional keyboard or dummy policy, no RoboCasa |

So: **robot assets**, **config**, **optional WBC policies**, and a **minimal “sim only” script** — same roles as sim2mujoco, but for the OmniGibson/BEHAVIOR stack.

---

## 3. Suggested directory layout

Place **sim2behavior** next to **sim2mujoco** under `gr00t_wbc/` so the pattern is consistent:

```
gr00t_wbc/
├── sim2mujoco/                    # existing
│   ├── resources/robots/g1/
│   ├── scripts/
│   └── ...
└── sim2behavior/                  # new
    ├── README.md
    ├── requirements.txt           # omnigibson, behavior, etc.
    ├── resources/
    │   └── robots/
    │       └── g1/
    │           ├── urdf/          # G1 URDF (OmniGibson often uses URDF)
    │           │   └── g1_29dof_with_hand.urdf
    │           ├── meshes/         # or link to sim2mujoco/meshes
    │           ├── g1_omnigibson_config.yaml   # controller + GR00T mapping
    │           └── (optional) policy/          # if reusing Balance/Walk in OmniGibson
    └── scripts/
        ├── run_omnigibson_g1_standalone.py     # minimal OmniGibson + G1
        └── (optional) run_omnigibson_g1_wbc.py # G1 + WBC ONNX in OmniGibson
```

Alternative: put **sim2behavior** under the main Isaac-GR00T repo (e.g. `gr00t/eval/sim/sim2behavior/`) if you want it to live with the rest of the BEHAVIOR/OmniGibson eval code. The design below works either way; only the import paths and `gr00t_wbc` vs `gr00t` references change.

---

## 4. Robot and assets

- **OmniGibson** typically uses **URDF** (and sometimes USD) for robots. You can:
  - **Reuse** the G1 URDF from **sim2mujoco** (`resources/robots/g1/g1.urdf`) and copy or symlink the **meshes** so OmniGibson finds them (paths in URDF must match).
  - Or add an OmniGibson-specific URDF in `sim2behavior/resources/robots/g1/urdf/` if the engine needs different conventions (e.g. different mesh paths, or a specific root).
- Keep a **single source of truth** for the mechanical model (e.g. sim2mujoco) and let sim2behavior reference or copy it so you don’t maintain two robot descriptions.

---

## 5. Config: `g1_omnigibson_config.yaml`

Define a config that plays the same role as `g1_gear_wbc.yaml` but for OmniGibson:

- **Robot:** path to URDF (or scene with robot), mesh dir.
- **Controllers:** OmniGibson controller names and parameters (e.g. base, waist, arms, hands) — aligned with the deployment guide’s “Controller Definition” (base, arms, 3-finger hands).
- **Observation:** which OmniGibson obs keys map to GR00T (video, state, language) and any scaling/history.
- **Action:** how GR00T action chunks (e.g. 30-step) are converted to OmniGibson commands (which joints, order, scale) — same idea as “Model-Controller Interface” in the deployment guide.

Example skeleton:

```yaml
# sim2behavior/resources/robots/g1/g1_omnigibson_config.yaml
robot:
  urdf_path: "urdf/g1_29dof_with_hand.urdf"
  meshes_dir: "meshes"

controllers:
  base: { type: "diff_drive", ... }
  waist: { ... }
  arms: { ... }
  hands: { type: "multi_finger", finger_type: "Dex3_1", ... }

gr00t_interface:
  obs_keys: ["rgb", "joint_state", "task_instruction"]
  action_execution_steps: 16
  action_mapping: { ... }
```

The existing **BEHAVIORGr00tEnv** (and any **UnitreeG1Dex3Env** from the deployment guide) can load this config so that observation preprocessing and action application stay in one place and are consistent with a standalone script.

---

## 6. Standalone script: “OmniGibson + G1 only”

Mirror **run_mujoco_gear_wbc.py** with an OmniGibson version:

- **Goal:** Run an OmniGibson environment that contains only (or mainly) the G1 robot — no full BEHAVIOR task, no GR00T server. Use it to:
  - Check that the G1 loads and is controllable in OmniGibson.
  - Optionally drive the robot with keyboard or a simple policy (e.g. zero action, or the same Balance/Walk ONNX if you add an OmniGibson WBC adapter).
- **Steps:**
  1. Depend on `omnigibson` (and `behavior` if you use BEHAVIOR tasks).
  2. Load scene config that spawns the G1 (from sim2behavior resources or a minimal scene).
  3. Create `Environment` (or equivalent) with robot from `g1_omnigibson_config.yaml`.
  4. Loop: `env.step(action)`; action from keyboard or from a small policy that reads `env` state and returns commands in OmniGibson format.
- **Location:** e.g. `sim2behavior/scripts/run_omnigibson_g1_standalone.py`.

This gives a “sim2behavior” entry point analogous to `run_mujoco_gear_wbc.py` for MuJoCo.

---

## 7. WBC (Balance/Walk) in OmniGibson (optional)

In **sim2mujoco**, the same ONNX policies run inside MuJoCo’s step loop. To mirror that in **sim2behavior**:

- **Option A – Full WBC in OmniGibson:**  
  In the standalone script (or a separate `run_omnigibson_g1_wbc.py`), each step:
  1. Read from OmniGibson the state needed for the existing observation builder (joint pos/vel, base orientation, commands, etc.).
  2. Build the 516-dim observation (or whatever the ONNX expects) and run Balance/Walk ONNX.
  3. Map the ONNX output (15-dim) to OmniGibson’s joint/actuator commands and apply them (e.g. via OmniGibson’s API or by setting targets for a low-level controller).

  This requires an **OmniGibson-side observation builder** that matches the one in sim2mujoco (and possibly the same `g1_gear_wbc.yaml` for obs/action dims and gains), and a small adapter that turns ONNX outputs into OmniGibson actions.

- **Option B – No WBC in sim2behavior:**  
  Use OmniGibson’s built-in controllers only; GR00T outputs high-level actions and the existing BEHAVIOR/OmniGibson wrapper converts them to controller commands (as in the deployment guide). Then sim2behavior only provides **robot + config + standalone script** without reusing the ONNX. You can add WBC later if needed.

Start with **Option B** to get a minimal sim2behavior; add Option A if you need the same locomotion stack in OmniGibson as in MuJoCo.

---

## 8. Integration with existing BEHAVIOR/OmniGibson code

- **gr00t/eval/sim/BEHAVIOR/** already has:
  - `BEHAVIORGr00tEnv` (OmniGibson + GR00T obs/action conversion),
  - `og_teleop_cfg.py` / `og_teleop_utils.py` (scene, tasks, simulator name),
  - Task list and registration for `sim_behavior_r1_pro/...` (Galaxea R1 Pro).

- **sim2behavior** should:
  - **Not replace** that code; instead, **feed** it (or a G1-specific env) with:
    - Robot URDF and meshes from `sim2behavior/resources/robots/g1/`.
    - Controller and GR00T-interface config from `g1_omnigibson_config.yaml`.
  - **Optional:** Define a **UnitreeG1Dex3Env** (as in the deployment guide) that subclasses `BEHAVIORGr00tEnv` and loads robot + config from sim2behavior (so one place owns G1-in-OmniGibson assets and config).
  - Register envs like `sim_omnigibson_unitree_g1/<task_name>` that use that env class and config.

So: **sim2behavior** = resources + config + standalone script; **BEHAVIOR/** = generic wrapper and task registration; **deployment guide** = how to plug G1 + Dex3_1 into that wrapper using sim2behavior assets.

---

## 9. Implementation steps (concise)

1. **Create repo layout:** Add `sim2behavior/` under `gr00t_wbc/` (or under `gr00t/eval/sim/`), with `resources/robots/g1/` and `scripts/`.
2. **Robot assets:** Add or symlink G1 URDF and meshes (reuse sim2mujoco where possible); fix paths for OmniGibson.
3. **Config:** Add `g1_omnigibson_config.yaml` (robot path, controllers, GR00T obs/action mapping).
4. **Standalone script:** Implement `run_omnigibson_g1_standalone.py` (load OmniGibson env with G1, step loop, keyboard or trivial policy).
5. **Optional WBC:** If you want Balance/Walk in OmniGibson, add an obs builder and ONNX→OmniGibson action adapter and a second script or mode.
6. **Integration:** From `BEHAVIORGr00tEnv` or `UnitreeG1Dex3Env`, load robot and config from sim2behavior; register `sim_omnigibson_unitree_g1/...` and document in the deployment guide.

---

## 10. Summary

| Item | sim2mujoco | sim2behavior |
|------|------------|--------------|
| Simulator | MuJoCo | OmniGibson (BEHAVIOR) |
| Robot format | MJCF/URDF + STL | URDF + meshes (or USD if needed) |
| Config | g1_gear_wbc.yaml | g1_omnigibson_config.yaml |
| Low-level policy | Balance/Walk ONNX in-process | Optional: same ONNX + adapter, or OmniGibson controllers only |
| Standalone script | run_mujoco_gear_wbc.py | run_omnigibson_g1_standalone.py |
| Used by | G1GearWbcPolicy, teleop | BEHAVIORGr00tEnv / UnitreeG1Dex3Env, deployment guide |

Making **sim2behavior** follow this structure keeps it “similar to sim2mujoco” while using BEHAVIOR/OmniGibson as the simulator and reusing the existing OmniGibson integration and deployment guide where possible.

---

## 11. Virtual environment setup (GR00T and BEHAVIOR)

Running GR00T (model server or training) and BEHAVIOR/OmniGibson (sim2behavior client, standalone scripts, or BEHAVIOR tasks) often involves **different Python stacks**. Use **separate virtual environments** to avoid dependency conflicts (e.g., different PyTorch/CUDA, omnigibson vs gr00t deps).

### 11.1 GR00T environment (Isaac-GR00T repo)

- **Purpose:** Model loading, GR00T server, training, and any code under `gr00t/` that does not import OmniGibson/BEHAVIOR.
- **Setup:** From the Isaac-GR00T repo root:
  - Create venv: `python -m venv .venv_gr00t` (or use `uv venv .venv_gr00t`).
  - Activate and install: `pip install -e .` (and any extras from `pyproject.toml`).
  - Use this when: running the GR00T server, launching training, or running eval clients that only talk to the server over the network (no local OmniGibson import).

### 11.2 BEHAVIOR / sim2behavior environment

- **Purpose:** OmniGibson, BEHAVIOR tasks, sim2behavior resources, and scripts that import `omnigibson` / `behavior` (e.g. `run_omnigibson_g1_standalone.py`, BEHAVIORGr00tEnv-based eval).
- **Setup:**
  - Create a dedicated venv, e.g. `python -m venv .venv_behavior` (or under BEHAVIOR repo / a shared env used for OmniGibson).
  - Install OmniGibson and BEHAVIOR per their official docs; install sim2behavior deps (e.g. `sim2behavior/requirements.txt` if present).
  - Optionally install only the GR00T **client** dependencies (e.g. `gr00t[client]` or minimal subset) if the BEHAVIOR process connects to a GR00T server and does not load the model locally.
- **Use this when:** Running the OmniGibson standalone script, BEHAVIOR task evaluation that connects to a GR00T server, or any script that imports `omnigibson` or `behavior`.

### 11.3 Two-venv workflow (server + client)

Typical evaluation flow:

1. **Terminal 1 (GR00T venv):** Activate `.venv_gr00t`, start the GR00T server (loads checkpoint, serves actions).
2. **Terminal 2 (BEHAVIOR venv):** Activate `.venv_behavior`, run the eval client (e.g. `rollout_policy.py` or a BEHAVIOR-specific runner) that connects to the server and runs the OmniGibson/BEHAVIOR environment.

No need to install OmniGibson in the GR00T venv or the full GR00T training stack in the BEHAVIOR venv; the client only needs the protocol (and optionally a small client library) to talk to the server.

### 11.4 Single-process (client loads model)

If the eval process **loads the GR00T model inside the same process** as OmniGibson (e.g. for debugging or simpler deployment), then that single process must have **both** GR00T and OmniGibson/BEHAVIOR dependencies. In that case use **one** environment where you install both stacks and ensure compatible versions (Python, PyTorch, CUDA). Prefer the two-venv (server + client) setup when possible to keep environments minimal and avoid conflicts.
