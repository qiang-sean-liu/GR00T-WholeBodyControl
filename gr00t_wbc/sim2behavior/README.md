# sim2behavior

OmniGibson/BEHAVIOR counterpart to **sim2mujoco**: robot assets, config, and scripts for the **Unitree G1** robot in OmniGibson (BEHAVIOR stack).

## Contents

- **resources/robots/g1/** – G1 URDF, meshes (symlink to sim2mujoco), and `g1_omnigibson_config.yaml` (controller + GR00T interface).
- **scripts/run_omnigibson_g1_standalone.py** – Minimal OmniGibson environment with G1 only (no full BEHAVIOR task). Use to verify G1 loads and is controllable.
- **scripts/run_omnigibson_g1_with_gr00t.py** – Run OmniGibson G1 with GR00T policy (uses observation and action adapters).
- **scripts/omnigibson_gr00t_adapters.py** – Observation and action adapters to connect GR00T policy with OmniGibson.

## Requirements

- Python 3.10+
- [OmniGibson](https://behavior.stanford.edu/omnigibson/) and optionally [BEHAVIOR](https://behavior.stanford.edu/) for task evaluation.
- Install deps: `pip install -r requirements.txt` (use a **dedicated venv** for BEHAVIOR/OmniGibson; see [sim2behavior_DESIGN.md](../sim2behavior_DESIGN.md) §11).
- **For GR00T integration:** The script automatically adds the Isaac-GR00T repo root to `sys.path` to import `gr00t` from the repo (located at `Isaac-GR00T/gr00t/`). You may need to install GR00T dependencies: `pip install 'numpy<2.0.0,>=1.23.5' msgpack pyzmq pin` (numpy version fix for OmniGibson compatibility, msgpack for serialization, pyzmq for ZMQ communication, pin/pinocchio for robot kinematics).

## Robot assets

- **URDF:** `resources/robots/g1/urdf/g1_29dof_with_hand.urdf` (derived from sim2mujoco’s `g1.urdf`, MuJoCo-specific tags removed).
- **Meshes:** `resources/robots/g1/meshes/` is a symlink to `sim2mujoco/resources/robots/g1/meshes/`. If missing, run:
  ```bash
  python3 scripts/setup_meshes_symlink.py
  ```
  Or manually: `cd resources/robots/g1 && ln -snf ../../../../sim2mujoco/resources/robots/g1/meshes meshes`
- **Config:** `resources/robots/g1/g1_omnigibson_config.yaml` – robot path, controllers (base, waist, arms, hands), and GR00T obs/action mapping.

## Standalone script (G1 only)

Activate your BEHAVIOR/OmniGibson environment first (e.g. `conda activate behavior`), then run from the **Isaac-GR00T** repo root:

```bash
python external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2behavior/scripts/run_omnigibson_g1_standalone.py
```

Or from `gr00t_wbc/sim2behavior`:

```bash
python scripts/run_omnigibson_g1_standalone.py
```

**Shell script (sets data path + optional video):** From `gr00t_wbc/sim2behavior` you can use the wrapper script, which exports `OMNIGIBSON_DATA_PATH` (so the G1 USD with cameras is used) and supports `--video` to dump frames:

```bash
./scripts/run_omnigibson_g1_standalone.sh              # run, no video
./scripts/run_omnigibson_g1_standalone.sh --video      # run and save video to output_frames/
./scripts/run_omnigibson_g1_standalone.sh -v           # same as --video
OMNIGIBSON_MAX_STEPS=50 ./scripts/run_omnigibson_g1_standalone.sh --video
```

**Optional environment variables:**

| Variable | Example | Description |
|----------|----------|--------------|
| `OMNIGIBSON_MAX_STEPS` | `100` | Max simulation steps (default: 1000). Use fewer steps for quick tests. |
| `OMNIGIBSON_SKIP_CLOSE` | `1` | Skip `env.close()` to avoid segfault on exit (exit 139); for debugging only. |
| `SAVE_VIDEO` | `1` | Save RGB frames to `sim2behavior/output_frames/` (requires `imageio` for MP4). |
| `DUMP_CAMERAS` | `1` | After first step, write **intrinsics and extrinsics** for all VisionSensors to `output_frames/camera_params.yaml`. Use to verify robot and external camera pose/FOV. See [docs/USD_CAMERAS_AND_CONFIG.md](docs/USD_CAMERAS_AND_CONFIG.md#robot-camera-extrinsics-and-intrinsics-where-they-come-from-and-how-to-verify). |

**Examples:**

```bash
# Short run (50 steps)
OMNIGIBSON_MAX_STEPS=50 python scripts/run_omnigibson_g1_standalone.py

# With display (viewport window)
DISPLAY=:0 OMNIGIBSON_HEADLESS=0 python scripts/run_omnigibson_g1_standalone.py

# Headless + save video
SAVE_VIDEO=1 python scripts/run_omnigibson_g1_standalone.py

# Avoid segfault on exit (skip teardown; for CI/debug)
OMNIGIBSON_SKIP_CLOSE=1 python scripts/run_omnigibson_g1_standalone.py
```

The script loads the G1 from sim2behavior resources, creates a minimal OmniGibson scene, and runs a step loop (zero action). Use it to confirm the G1 model and meshes load correctly in OmniGibson before running full BEHAVIOR tasks. If you get a segmentation fault (exit 139), it often occurs in `env.close()`; see the script docstring and the optional env vars above.

### Driving G1 with the GR00T model

**In OmniGibson:** Use `run_omnigibson_g1_with_gr00t.py` to run G1 with the GR00T policy (observation and action adapters). See [docs/RUNNING_G1_WITH_GR00T.md](docs/RUNNING_G1_WITH_GR00T.md) for command lines.

**In MuJoCo (recommended for evaluation):** Use the **MuJoCo-based** evaluation: run the GR00T server with `--embodiment-tag UNITREE_G1`, then run the rollout client with an env name like `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`. See [docs/RUNNING_G1_WITH_GR00T.md](docs/RUNNING_G1_WITH_GR00T.md) for step-by-step commands and [examples/GR00T-WholeBodyControl/EVALUATION_G1_WBC.md](../../../../examples/GR00T-WholeBodyControl/EVALUATION_G1_WBC.md) for full evaluation setup.

**Why is `obs["unitree_g1"]` only `{"proprio"}` (no `"rgb"`)?** The URDF has camera **links** (`d435_link`, `mid360_link`), but the standalone uses the built-in **UnitreeG1**, which loads from the **USD** in `omnigibson-robot-assets`, not from this URDF. OmniGibson discovers vision sensors from **Camera prims** (UsdGeom.Camera) as **children** of link prims in that USD. Standard URDF only defines links (no Camera prims); the BEHAVIOR-1K import script adds Camera prims when building the G1 USD if `camera_links: ["d435_link", "mid360_link"]` is set. If your installed G1 USD was built without that step, it has no Camera prims and obs will only have proprio. **To use a G1 USD that already has cameras** (e.g. under `.../BEHAVIOR-1K-datasets/omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda`): set **`OMNIGIBSON_DATA_PATH`** to the dataset root (parent of `omnigibson-robot-assets`) before importing OmniGibson, then run the script; see [docs/USD_CAMERAS_AND_CONFIG.md](docs/USD_CAMERAS_AND_CONFIG.md) for the exact path and robot config.

### OmniGibson: defining cameras and which are used for observations

**1. Defining cameras**

- **On the robot (USD):** Cameras are **Camera prims** (UsdGeom.Camera) under link prims in the robot’s USD. They are **not** defined in URDF. To get them:
  - **Import script:** When building the robot USD (e.g. BEHAVIOR-1K’s `import_custom_robot.py`), set **`camera_links`** in the import config. Each entry can be a link name (e.g. `d435_link`) or a dict: `link`, optional `parent_link`, and `offset` (position/orientation). The script adds a `Camera` prim as a child of that link (with optional offset). Example (full format): `camera_links: [{ link: "d435_link", parent_link: null, offset: { position: [0,0,0], orientation: [0,0,0,1] } }, ...]`. For G1, `camera_links: ["d435_link", "mid360_link"]` is used so the resulting USD has Camera prims under those links.
  - **External (scene) cameras:** In the **env config** (e.g. `g1_standalone.yaml`), you can add **`external_sensors`**: a list of sensor configs (type, name, position, orientation, …). Each sensor is created in the scene (not on the robot). Use `create_sensor`-compatible kwargs (e.g. VisionSensor with modalities). Set **`include_in_obs: true`** for any sensor whose output should appear in `obs["external"]`. `env.render()` returns RGB from external VisionSensors.

**2. Specifying which camera(s) are used for observations**

- **Robot obs:** Only prims whose **type** is `Camera` (or `Lidar`) under the robot’s links are turned into sensors; they are discovered at load time. Then:
  - **`obs_modalities`** (robot config): Include `"rgb"` (or `"all"`) so that vision sensors are given modalities (rgb, depth, etc.). If you omit rgb, vision sensors are still created but their modalities may be empty and they won’t contribute to obs.
  - **`include_sensor_names`** / **`exclude_sensor_names`** (robot config): Substrings matched against the **sensor prim path**. Only sensors whose prim path (a) does not contain any `exclude_sensor_names` and (b) either has no `include_sensor_names` or contains one of them are loaded. So you can restrict to e.g. `d435` or `mid360` by setting `include_sensor_names: ["d435_link"]` to use only cameras under that link.
  - **`sensor_config`** (robot config): Under `sensor_config["VisionSensor"]` you can set **`sensor_kwargs`**: e.g. **`image_height`**, **`image_width`**, **`focal_length`**, **`modalities`** (e.g. `["rgb"]` or `["rgb", "depth_linear"]`). This applies to **all** VisionSensors on that robot (no per-camera key in the current API). So you choose resolution and modalities for every onboard camera together.

- **External obs:** For sensors in **`external_sensors`**, each entry can set **`include_in_obs: true/false`**. Only those with `true` appear under `obs["external"][sensor_name]`.

**Summary**

| Goal | Where | What to set |
|------|--------|-------------|
| Add cameras on the robot | Import config (when building USD) | `camera_links: [link names or list of {link, parent_link, offset}]` |
| Add a scene camera | Env config | `external_sensors: [{ type: "VisionSensor", name: "...", position: [...], include_in_obs: true, ... }]` |
| Use rgb in robot obs | Robot config | `obs_modalities: ["proprio", "rgb"]` (and ensure USD has Camera prims) |
| Restrict which robot cameras are loaded | Robot config | `include_sensor_names: ["d435_link"]` or `exclude_sensor_names: ["mid360"]` |
| Set resolution/modalities for robot cameras | Robot config | `sensor_config: { VisionSensor: { sensor_kwargs: { image_height: 224, image_width: 224, modalities: ["rgb"] } } }` |

**Example: top-down external camera** – To add a scene camera that looks at the robot from above (sees full body), add **`env.external_sensors`** with a VisionSensor. `configs/g1_standalone.yaml` includes a `topdown_camera`: position `[0, 0, 2.8]` (above the robot at z≈0.8), orientation **`[0, 0, 0, 1]`** (identity; in Isaac Sim Z-up, the camera looks along -Z = straight down). Use identity for top-down; do *not* use `[0.707, 0, 0, 0.707]` (that points the camera along +Y, so the robot is not in view). Its images appear in `obs["external"]["topdown_camera"]`. Adjust `position[2]` or `sensor_kwargs.focal_length` (smaller = wider FOV) if the robot is not fully in frame.

## Integration with BEHAVIOR/GR00T

- **BEHAVIORGr00tEnv** (in `gr00t/eval/sim/BEHAVIOR/`) can load robot and config from sim2behavior for G1-based evaluation.
- See [sim2behavior_DESIGN.md](../sim2behavior_DESIGN.md) and the [GR00T OmniGibson Deployment Guide](../../../GR00T_OmniGibson_Deployment_Guide.md) for architecture and virtual environment setup (GR00T vs BEHAVIOR venvs).
