# Using a custom Unitree G1 USD with cameras (sim2behavior)

## Verification: cameras in the USD

The USD at:

```
/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda
```

**contains two Camera prims:**

- Under **`d435_link`**: `def Camera "Camera"` (child of `def Xform "d435_link"`)
- Under **`mid360_link`**: `def Camera "Camera"` (child of `def Xform "mid360_link"`)

OmniGibson discovers vision sensors by scanning link prims for children of type **Camera**; this USD therefore provides two VisionSensors (d435 and mid360) once the robot is loaded from this asset path.

The same layout exists in `import_config.yaml` in that directory (`camera_links: [d435_link, mid360_link]`).

---

## How to use this USD with sim2behavior

UnitreeG1 in OmniGibson resolves its USD path as:

```text
get_dataset_path("omnigibson-robot-assets") + "models/unitree_g1/usd/unitree_g1.usda"
```

`get_dataset_path("omnigibson-robot-assets")` equals **`$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets`** (see `omnigibson/macros.py`). So the **dataset root** must be the parent of the `omnigibson-robot-assets` folder.

### 1. Point OmniGibson at the dataset that contains this USD

Set **before** any `import omnigibson` (e.g. in the script or in the shell). Default in the run script is `/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets`:

```bash
export OMNIGIBSON_DATA_PATH=/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets
```

Then:

- `get_dataset_path("omnigibson-robot-assets")` → `.../datasets/omnigibson-robot-assets`
- UnitreeG1 loads `.../omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda` (the USD with cameras).

### 2. Run the standalone script with that env set

```bash
export OMNIGIBSON_DATA_PATH=/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets
python scripts/run_omnigibson_g1_standalone.py
```

Or from the Isaac-GR00T repo root:

```bash
export OMNIGIBSON_DATA_PATH=/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets
python external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2behavior/scripts/run_omnigibson_g1_standalone.py
```

The script does **not** set `OMNIGIBSON_DATA_PATH` by default; you must set it so that the Unitree G1 USD with cameras is the one that gets loaded.

### 3. Robot config for vision (already in `g1_standalone.yaml`)

To use the onboard cameras for observations, the robot config must request **rgb** and can tune the VisionSensor. The existing `configs/g1_standalone.yaml` is already set up for that:

```yaml
robots:
  - type: "UnitreeG1"
    name: "unitree_g1"
    obs_modalities: ["proprio", "rgb"]   # include rgb so Camera prims become VisionSensors with rgb
    sensor_config:
      VisionSensor:
        sensor_kwargs:
          image_height: 224
          image_width: 224
          # modalities: ["rgb"]  # optional; defaults from obs_modalities
```

- **`obs_modalities: ["proprio", "rgb"]`** – Ensures vision sensors get the `rgb` modality and appear in `obs["unitree_g1"]`.
- **`sensor_config.VisionSensor.sensor_kwargs`** – Applies to **all** VisionSensors on the robot (both d435 and mid360). Resolution is shared; there is no per-camera key in the current API.

Optional filters (if you want only one camera in obs):

- **`include_sensor_names: ["d435_link"]`** – Only sensors whose prim path contains `d435_link` are loaded (e.g. only the d435 camera).
- **`exclude_sensor_names: ["mid360_link"]`** – Exclude the mid360 camera.

Example: use only the d435 camera:

```yaml
robots:
  - type: "UnitreeG1"
    name: "unitree_g1"
    obs_modalities: ["proprio", "rgb"]
    include_sensor_names: ["d435_link"]
    sensor_config:
      VisionSensor:
        sensor_kwargs:
          image_height: 224
          image_width: 224
```

---

## Robot camera extrinsics and intrinsics: where they come from and how to verify

### Where extrinsics come from

**Robot onboard cameras:** The **extrinsics** (camera pose in world or robot frame) come from the **USD scene graph**. Each camera is a **Camera prim** that is a **child of a link** (e.g. `d435_link`, `mid360_link`). The camera’s pose in the world is:

- **Link pose** (from the robot’s current state: base + joints) × **Camera offset** (the Camera prim’s transform relative to its parent link in the USD).

So if the robot sees “nothing meaningful,” common causes are:

1. **Wrong or default offset in the USD** – The Camera prim may have identity (or zero) transform relative to the link, or the wrong orientation (e.g. looking into the robot body). The G1 USD must have been built with correct `camera_links` (and optional `offset` position/orientation) in the import config so that the Camera is placed and oriented correctly on the head/body.
2. **Wrong link pose** – Less common if the robot USD is correct; the link pose is driven by kinematics.

**External (scene) cameras:** Extrinsics come from the env config: `external_sensors[].position`, `orientation`, and `pose_frame` (e.g. `scene`). See the top-down camera example in `g1_standalone.yaml`.

### Where intrinsics come from

- **Focal length / FOV:** Set by the **VisionSensor** when the Camera prim is created or updated. OmniGibson’s VisionSensor uses `sensor_kwargs.focal_length` (default 17 mm) and the USD Camera prim’s **horizontalAperture** (and related) if present. So:
  - **Robot cameras:** You can override for all robot VisionSensors via `sensor_config.VisionSensor.sensor_kwargs.focal_length` in the robot config (e.g. in `g1_standalone.yaml`). If not set, the default (or whatever is in the USD) is used.
  - **External cameras:** Set in `external_sensors[].sensor_kwargs.focal_length`.
- **Resolution:** `sensor_kwargs.image_width` and `image_height` (robot: under `sensor_config.VisionSensor.sensor_kwargs`; external: under each sensor’s `sensor_kwargs`).

### How to verify: dump all camera parameters

Run the standalone script with **`DUMP_CAMERAS=1`** (and optionally `SAVE_VIDEO=1`). After the first sim step, the script writes **`output_frames/camera_params.yaml`** with, for each VisionSensor (robot and external):

- **`prim_path`** – USD path of the camera prim.
- **`focal_length_mm`** – Focal length in mm.
- **`camera_view_transform_world_from_camera_4x4`** – 4×4 view matrix (world-from-camera); use this to check pose (position and orientation).
- **`intrinsic_matrix_3x3`** – 3×3 intrinsic matrix (fx, fy, cx, cy, etc.).
- **`clipping_range`**, **`aperture`** – For debugging near/far and FOV.

Example:

```bash
DUMP_CAMERAS=1 SAVE_VIDEO=1 python scripts/run_omnigibson_g1_standalone.py
# Then inspect: output_frames/camera_params.yaml
```

Check that:

- **Robot cameras:** The view transform places the camera on the robot (e.g. near the head) and the third column (or your convention’s “forward” axis) points **outward** from the robot (e.g. forward or down for a head camera). If the camera is inside the mesh or pointing backward, the USD Camera offset (or link) is wrong.
- **Focal length:** If the image is too narrow (only a small patch of scene), decrease `focal_length` in the relevant `sensor_kwargs` (or in robot `sensor_config.VisionSensor.sensor_kwargs` for onboard cams).

### How to fix robot cameras that see nothing meaningful

1. **Re-export the G1 USD with correct camera placement**  
   When building the robot USD (e.g. BEHAVIOR-1K `import_custom_robot.py` or equivalent), set **`camera_links`** with the correct **`offset`** (position and orientation in the link frame) so that each Camera prim is in front of the face/sensor and looking outward. Example (conceptual):

   ```yaml
   camera_links:
     - link: d435_link
       offset:
         position: [0.02, 0, 0.05]   # in link frame; adjust to match real mount
         orientation: [0, 0, 0, 1]   # or rotate so camera looks forward
     - link: mid360_link
       offset: { position: [0, 0, 0], orientation: [0, 0, 0, 1] }
   ```

   Then point **`OMNIGIBSON_DATA_PATH`** at the dataset that contains this updated `omnigibson-robot-assets` so UnitreeG1 loads this USD.

2. **Tune intrinsics in config (without changing USD)**  
   In `g1_standalone.yaml` (or your env config), under the robot’s **`sensor_config.VisionSensor.sensor_kwargs`**, set **`focal_length`** (e.g. `12.0` for wider FOV). This does not fix wrong **extrinsics** (pose); if the camera is inside the mesh or pointing the wrong way, you must fix the USD as above.

3. **Sanity check with DUMP_CAMERAS**  
   After any change, run again with `DUMP_CAMERAS=1` and inspect `camera_params.yaml` to confirm view transforms and intrinsics.

---

## Validating the D435 camera on G1

### 1. Position: which coordinate system?

The D435 camera’s **position** is **not** stored in the env config. It comes from the **USD scene graph** in **world coordinates** (Isaac Sim convention: **Z-up**, **+X forward**, right-handed).

- The camera is a **Camera prim** under the link **`d435_link`** in the robot USD.
- Its **world pose** each frame is: **robot base pose** (world) × **joint chain** to `d435_link` × **Camera offset** (transform of the Camera prim relative to `d435_link` in the USD).
- So the position is in **world frame** (meters). The **Camera offset** is defined in the **link frame** of `d435_link` when the G1 USD was built (e.g. via `camera_links` and `offset` in the import config).

To inspect the actual world pose (and intrinsics), run with **`DUMP_CAMERAS=1`** (see below); the dump contains **`camera_view_transform_world_from_camera_4x4`** for each sensor.

### 2. How is the scene projected onto the D435?

- OmniGibson attaches a **VisionSensor** to the Camera prim at `prim_path` (e.g. under `/World/.../unitree_g1/d435_link/Camera`).
- Each time you get an observation, the sensor calls **Isaac Sim’s renderer** with that prim as the active camera. The renderer uses:
  - **Extrinsics:** the Camera prim’s **current world transform** (from the scene graph: robot pose × link pose × camera offset).
  - **Intrinsics:** focal length (from `sensor_config.VisionSensor.sensor_kwargs.focal_length` or the USD), **horizontalAperture** (from USD or defaults), and **image_width** / **image_height** (from config). The scene is projected with a standard **perspective projection** and **clipping_range** (near/far).
- So the “projection” is: **world → camera frame** via the view matrix, then **perspective project** using the 3×3 intrinsic matrix. No separate “scene-to-camera” step; the scene is already in world frame and the camera pose is in world frame.

### 3. Why are D435 images static and half black?

**Static (very little motion):**

- The camera pose **does** update every step with the robot (it’s under `d435_link` in the scene graph). If the video looks static, the usual cause is that the **robot is barely moving** (e.g. policy outputs near-zero or constant actions, or the robot is standing still). Check that the robot is actually moving in the sim (e.g. top-down or external camera video).
- Less often: the same camera image is being reused (e.g. a bug in observation collection). Verify that `og_obs` changes each step (e.g. log a norm of the d435 rgb at step 0 and 100).

**Half the image pure black:**

- **Wrong camera orientation in the USD:** If the Camera prim’s **offset** relative to `d435_link` is identity or points **into** the robot, half the image can be the robot body (dark) or inside the mesh (black). Fix by re-exporting the G1 USD with correct **`camera_links`** and **`offset`** (position and orientation in the link frame) so the camera looks **outward** (e.g. forward in the head frame).
- **Camera inside geometry:** If the offset places the camera **inside** the head or body mesh, you get clipping or black regions. Move the camera **offset** forward (or in the correct direction) so the camera is clearly outside the mesh.
- **Clipping / FOV:** A very large **near** clip or wrong **clipping_range** can clip the scene. Check **`clipping_range`** in the camera dump; typical values are (e.g. 0.01, 100) in meters.

**What to do:**

1. Run with **`DUMP_CAMERAS=1`** (and optionally **`--save_video`**) so you get **`output_frames/camera_params.yaml`** and D435 video. For the GR00T run script, set the env var:  
   `DUMP_CAMERAS=1 python scripts/run_omnigibson_g1_with_gr00t.py ... --save_video`
2. In **`camera_params.yaml`**, find the D435 sensor (e.g. `unitree_g1:d435_link:Camera:0`). Check **`camera_view_transform_world_from_camera_4x4`**: the camera should be in front of the head and the “forward” axis should point **out** of the robot, not into it.
3. If the transform is wrong, fix the **G1 USD**: re-run the import with **`camera_links`** and an **`offset`** that places the Camera in front of the face and looking forward (or the desired direction). Then point **`OMNIGIBSON_DATA_PATH`** at the dataset that contains this updated USD.

---

## G1 evaluation camera parameters (aligned with sim2behavior)

For full details on how GR00T G1 evaluation is done, which camera feeds the model, and how to match OmniGibson USD to MuJoCo, see **[GR00T_G1_EVALUATION_AND_CAMERAS.md](GR00T_G1_EVALUATION_AND_CAMERAS.md)**.

GR00T G1 evaluation (SyncEnv + rollout_policy) uses these camera constants from **`gr00t_wbc/data/constants.py`**:

- **RS_VIEW_CAMERA_WIDTH = 640**
- **RS_VIEW_CAMERA_HEIGHT = 480**

Camera names in evaluation: `robot0_oak_egoview`, `robot0_rs_tppview` (MuJoCo/RoboCasa). RoboCasa agentview cameras use **fovy = 60°** (see [ROBOCASA_CAMERA_PARAMS.md](ROBOCASA_CAMERA_PARAMS.md)).

**`configs/g1_standalone.yaml`** is set to use the same resolution (640×480) and a similar FOV (focal_length 12) for both the robot VisionSensors and the topdown external camera, so sim2behavior images match the evaluation setup where possible.

---

## Summary

| Step | Action |
|------|--------|
| 1 | Set `OMNIGIBSON_DATA_PATH` to the **parent** of `omnigibson-robot-assets` (e.g. `.../BEHAVIOR-1K/datasets`) **before** importing OmniGibson. |
| 2 | Run the standalone script (or any env that uses UnitreeG1) without changing code; UnitreeG1 will load `omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda` from that path. |
| 3 | Use `configs/g1_standalone.yaml` (or the same robot config): `obs_modalities: ["proprio", "rgb"]` and optional `sensor_config.VisionSensor.sensor_kwargs` and `include_sensor_names` / `exclude_sensor_names` as above. |

With this, `obs["unitree_g1"]` will contain vision data from the Camera prims in that USD (e.g. under sensor names like `unitree_g1:d435_link:Camera:0` and `unitree_g1:mid360_link:Camera:0` in the nested obs dict).
