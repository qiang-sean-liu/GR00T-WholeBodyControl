# GR00T G1 Evaluation Setup and Cameras

This document summarizes how GR00T model evaluation is done with the Unitree G1 robot (with Dex3-style 3-finger hands), how cameras are configured in MuJoCo evaluation, and how to align OmniGibson (sim2behavior) with those settings.

---

## 1. How GR00T G1 evaluation is done (G1 + Dex3-style hands)

### 1.1 Evaluation flow

- **Server:** `gr00t/eval/run_gr00t_server.py` with `--embodiment-tag UNITREE_G1` and `--use-sim-policy-wrapper`.
- **Client:** `gr00t/eval/rollout_policy.py` using the **GR00T-WholeBodyControl** venv. It creates a gym environment with:
  - `env_name` such as `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc`
  - `camera_names=["robot0_oak_egoview", "robot0_rs_tppview"]`
  - WBC wrapper with controller config `default_mink_ik_g1_gear_wbc.json`

Evaluation runs in **MuJoCo/RoboCasa** (SyncEnv + `Gr00tLocomanipRoboCasaEnv`), not in OmniGibson.

### 1.2 Robot configuration (G1 + 3-finger hands)

- **Robot model:** G1 from **dexmg/gr00trobocasa**:  
  `robocasa/models/assets/robots/unitree_g1/g1_29dof_rev_1_0.xml`
- **Hands:** **G1ThreeFingerLeftHand** and **G1ThreeFingerRightHand** (3-finger dexterous, Dex3-style).  
  Set in `G1.default_gripper` in  
  `gr00t_wbc/dexmg/gr00trobocasa/robocasa/models/robots/manipulators/g1_robot.py`.  
  There is no separate “Dex3_1” config; the 3-finger hand is the default G1 hand in this stack.
- **Control:** Whole-body control via **gear_wbc** (controller config `default_mink_ik_g1_gear_wbc.json`).

### 1.3 Camera configuration in MuJoCo (g1_robot.py)

Cameras are defined in **`G1.get_camera_configs()`** in  
`gr00t_wbc/dexmg/gr00trobocasa/robocasa/models/robots/manipulators/g1_robot.py`.  
They are virtual MuJoCo cameras (pos/quat relative to a parent body).

| Camera key             | Parent body   | Role in eval        | pos (m)                                              | quat (w,x,y,z)                                        | fovy   |
|------------------------|---------------|----------------------|------------------------------------------------------|--------------------------------------------------------|--------|
| robot0_oak_egoview     | torso_link    | **Model input**      | [0.10209156, -0.00937542, 0.42446595]               | [0.64367383, 0.26523914, -0.27106013, -0.66472446]    | 79.5°  |
| robot0_rs_tppview      | pelvis        | Logging / video only | [-1.131, -0.626, 0.454]                             | [0.67953146, 0.46872971, -0.3204774, -0.46456828]     | 60°    |
| robot0_rs_egoview      | torso_link    | Not used in rollout  | [0.07555294, 0.02579919, 0.49719054]                | [0.67876761, 0.20864099, -0.20567003, -0.67338199]   | 40°    |

### 1.4 Resolution and constants

- **gr00t_wbc/data/constants.py:**  
  `RS_VIEW_CAMERA_WIDTH = 640`, `RS_VIEW_CAMERA_HEIGHT = 480`  
  Used so sim2behavior can match evaluation resolution (640×480) where applicable.

---

## 2. Which camera feeds the GR00T model, and what is the other for?

### 2.1 Camera that generates input to the GR00T model

- **Policy modality (unitree_g1):** `gr00t/configs/data/embodiment_configs.py` sets  
  `modality_keys=["ego_view"]` for the video modality.  
  So the GR00T model receives **only one image**: **ego_view**.
- **Env → policy mapping:**  
  `gr00t_wbc/control/envs/robocasa/utils/cam_key_converter.py` (CameraKeyMapper) maps:
  - `robot0_oak_egoview` → **ego_view**
  - `robot0_rs_tppview` → **tpp_view**
- **Conclusion:** The image that generates the input to the GR00T model is **robot0_oak_egoview** (OAK-D ego view). In MuJoCo evaluation this camera is mounted on **torso_link** with the pos/quat/fovy given in the table above.

### 2.2 The other camera (robot0_rs_tppview)

- **robot0_rs_tppview** is **not** in the policy’s video modality keys for unitree_g1 (only `ego_view` is).
- It is used for:
  - **Logging and video recording** (e.g. third-person view in saved videos)
  - **Debugging and visualization** (e.g. default render camera in some code paths)
- So it does **not** feed the GR00T model; it is for observation and analysis only.

---

## 3. Matching OmniGibson USD to MuJoCo camera settings

In sim2behavior, the D435 camera in the G1 USD is mapped to **robot0_oak_egoview** (and thus to **ego_view**). To match MuJoCo’s evaluation view as closely as possible:

### 3.1 Which USD to edit

- **File:** The USD that OmniGibson actually loads:  
  `{OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda`  
  (path under dataset root: `models/unitree_g1/usd/unitree_g1.usda` — **no** extra `unitree_g1` in the path).

### 3.2 What to match

1. **Ego view (D435 in USD → oak_egoview)**  
   MuJoCo’s **robot0_oak_egoview** is defined relative to **torso_link**:
   - pos = [0.10209156, -0.00937542, 0.42446595] (m)
   - quat = [0.64367383, 0.26523914, -0.27106013, -0.66472446] (w,x,y,z)
   - fovy = 79.5°

   In the USD you have **d435_link** (with its own pose relative to the robot) and a **Camera** prim under it. The **combined** transform (d435_link in world × Camera in d435_link) should produce a similar view to the MuJoCo ego camera. You can:
   - Either adjust the **Camera** prim under `d435_link` (translate/orient) so that the resulting view matches the MuJoCo ego view, or
   - Keep the current working D435 fix (translate = (0,0,0), orient = (-0.5,-0.5,0.5,0.5)) and only match resolution/FOV if needed.

2. **Resolution**  
   Set VisionSensor (or equivalent) to **640×480** to match `RS_VIEW_CAMERA_WIDTH/HEIGHT` and MuJoCo eval.

3. **FOV (optional)**  
   MuJoCo oak_egoview uses **fovy = 79.5°**. In USD, you can set the Camera’s horizontal aperture / focal length so the vertical FOV is about 79.5° if you want pixel-level alignment.

### 3.3 Where to edit the USD

- Under `def Xform "d435_link"`, find `def Camera "Camera"`.
- Edit:
  - `double3 xformOp:translate` — position of camera in d435_link frame.
  - `quatd xformOp:orient` — orientation in d435_link frame.
- Resolution and FOV are typically set in the **robot/sensor config** (e.g. sim2behavior `configs/g1_standalone.yaml` or `g1_with_scene.yaml`) via `sensor_config.VisionSensor` and `image_width` / `image_height`; the USD defines the Camera prim’s intrinsics (e.g. focal length) if the pipeline uses them.

### 3.4 Reference: current working D435 extrinsics (sim2behavior)

After the fix described in `D435_USD_ANALYSIS_AND_FIX.md`, the Camera under `d435_link` uses:

- `double3 xformOp:translate = (0, 0, 0)`
- `quatd xformOp:orient = (-0.5, -0.5, 0.5, 0.5)`

This gives a correct forward-looking view with correct in-plane rotation. To align more closely with MuJoCo’s oak_egoview, you would need to compute the equivalent translate/orient in the d435_link frame such that the combined pose (in torso or world) matches the MuJoCo ego camera pose above.

---

## 4. Unitree G1 class in BEHAVIOR-1K and camera configuration

### 4.1 Location of the Unitree G1 class

- **File:** `/mnt/nas26/qiang.liu/BEHAVIOR-1K/OmniGibson/omnigibson/robots/unitree_g1.py`
- **Class:** `UnitreeG1(ManipulationRobot, LocomotionRobot, ActiveCameraRobot)`

### 4.2 Can we set cameras there like in g1_robot.py?

**No.** In BEHAVIOR-1K/OmniGibson, camera pose and intrinsics are **not** defined in the Python robot class.

- **MuJoCo (g1_robot.py):**  
  `G1.get_camera_configs()` returns a dictionary of camera names to `pos`, `quat`, `parent_body`, and `camera_attribs`. The environment uses this to **place** virtual cameras in the scene. So in MuJoCo, camera extrinsics are set in code.

- **OmniGibson (UnitreeG1):**  
  Vision sensors are discovered by scanning the **USD** for Camera prims under the robot’s link prims. The base class `robot_base.py` uses `_load_sensors()` and does **not** call any `get_camera_configs()`-style API.  
  `UnitreeG1` only defines:
  - `camera_link_names` → `["d435_link", "mid360_link"]` (used for which links may have cameras; the actual discovery is still by scanning the USD for Camera prims under those links).
  - No `get_camera_configs()` method.

So in OmniGibson, camera **pose and intrinsics** come from the **USD asset** (the Camera prim’s xform and attributes under `d435_link` / `mid360_link`). To change camera settings for the G1 in BEHAVIOR-1K you must:

1. **Edit the USD** (extrinsics and, if used, intrinsics), and/or  
2. **Use robot/sensor config** (e.g. in the scene or sim2behavior config YAML) for resolution, FOV, and which sensors are included (`include_sensor_names`, `sensor_config.VisionSensor`, etc.).

You cannot set camera pos/quat from `UnitreeG1` in the same way as in `g1_robot.get_camera_configs()`; the architecture is asset-driven, not code-driven for camera pose.

---

## 5. Summary table

| Item | MuJoCo evaluation | OmniGibson (sim2behavior) |
|------|--------------------|----------------------------|
| Robot | G1, `g1_29dof_rev_1_0.xml` (dexmg/gr00trobocasa) | UnitreeG1, USD from `models/unitree_g1/usd/unitree_g1.usda` |
| Hands | G1ThreeFingerLeftHand / RightHand (Dex3-style) | Same (3-finger hands in USD/URDF) |
| Camera that feeds GR00T | **robot0_oak_egoview** → ego_view | D435 Camera in USD mapped to robot0_oak_egoview → ego_view |
| Other camera | robot0_rs_tppview (logging/video) | mid360_link Camera → robot0_rs_tppview |
| Where camera pose is set | `g1_robot.get_camera_configs()` | USD (Camera prim under d435_link / mid360_link) |
| Resolution | 640×480 (constants.py) | Set in config YAML (e.g. 640×480 in g1_standalone.yaml) |

---

## 6. MuJoCo oak_egoview vs OmniGibson D435: frames, exact match, and FOV

### Quick reference: your four questions

1. **In robot0_oak_egoview, are pos and orient relative to torso_link?**  
   **Yes.** `parent_body` is `torso_link`; pos (m) and quat (w,x,y,z) are in the **torso_link** frame.

2. **In the OmniGibson USD, which link is the parent of the D435 camera, and what are the relative pose and orient?**  
   **Parent:** **d435_link** (the Camera prim is under `def Xform "d435_link"`).  
   **d435_link in torso_link (from URDF `d435_joint`):**  
   - `origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"`  
   So **p_torso_d435** = (0.0576235, 0.01753, 0.42987) m, **R_torso→d435** = R_Y(0.8308).  
   **Camera relative to d435_link (current working fix in USD):**  
   - `xformOp:translate = (0, 0, 0)`  
   - `xformOp:orient = (-0.5, -0.5, 0.5, 0.5)` (USD quat: x,y,z,w)

3. **To match MuJoCo robot0_oak_egoview, translate and orient in USD (in d435_link frame):**  
   Target in torso: **pos_torso** = [0.10209156, -0.00937542, 0.42446595], **quat_torso** = [0.64367383, 0.26523914, -0.27106013, -0.66472446] (w,x,y,z).  
   - **t_cam** = R_Y(−0.8308) · (pos_torso − p_torso_d435) → **(0.033976, −0.026905, 0.029194)** m  
   - **q_cam** raw then 180° roll so image is right-side up: **(−0.50776, −0.51093, 0.47955, 0.50116)** (x,y,z,w).  
   So in the USD Camera under d435_link set:
   ```usda
   double3 xformOp:translate = (0.033976, -0.026905, 0.029194)
   quatd xformOp:orient = (-0.5077606323, -0.510929544, 0.4795505526, 0.5011600631)
   ```

4. **Match other parameters (FOV):**  
   MuJoCo **fovy = 79.5°**. In config set **focal_length: 5.41** (mm) in `sensor_config.VisionSensor.sensor_kwargs` for the robot (with horizontalAperture 12 mm, verticalAperture 9 mm for 640×480). Resolution: **image_width: 640**, **image_height: 480**.

---

### 6.1 Is robot0_oak_egoview pose relative to torso_link?

**Yes.** In `g1_robot.get_camera_configs()`, `robot0_oak_egoview` has `parent_body=f"{self.naming_prefix}torso_link"`. So **pos** and **quat** are expressed in the **torso_link** frame (origin at torso_link, axes aligned with that link).

### 6.2 In the OmniGibson USD, what is the parent of the D435 camera and what is its relative pose?

- **Parent of the D435 camera:** The Camera prim is a **child of `d435_link`** (under `def Xform "d435_link"`). So the **parent body** of the D435 camera in the USD is **d435_link**, not torso_link.
- **d435_link’s own pose:** In the URDF, `d435_joint` attaches `d435_link` to **torso_link** with:
  - `origin xyz="0.0576235 0.01753 0.42987" rpy="0 0.8307767239493009 0"`  
  So in **torso_link** frame, d435_link origin is at (0.0576235, 0.01753, 0.42987) m with a rotation of ~0.83 rad (~47.6°) around the Y axis.
- **Current Camera pose relative to d435_link (in USD):**
  - `double3 xformOp:translate = (0, 0, 0)` — camera at d435_link origin.
  - `quatd xformOp:orient = (-0.5, -0.5, 0.5, 0.5)` — forward look + in-plane roll correction (from the earlier fix).

### 6.3 Computing translate and orient to match MuJoCo robot0_oak_egoview

We want the **combined** camera pose in **torso_link** frame to equal MuJoCo’s oak_egoview:

- **Target in torso_link:**  
  - **pos_torso** = [0.10209156, -0.00937542, 0.42446595] m  
  - **quat_torso** = [0.64367383, 0.26523914, -0.27106013, -0.66472446] (w,x,y,z)

- **d435_link in torso_link (from URDF d435_joint):**  
  - **p_torso_d435** = (0.0576235, 0.01753, 0.42987) m  
  - **R_torso→d435** = rotation by 0.8307767239493009 rad around Y (pitch only).

- **Camera in d435_link:** translate **t_cam**, orient **q_cam**. In torso:  
  - **p_cam_torso** = R_torso→d435 · **t_cam** + **p_torso_d435**  
  - **R_cam_torso** = R_torso→d435 · R_cam_d435  

  So we need:
  - **t_cam** = R_torso→d435^T · (p_cam_torso − p_torso_d435)  
  - **R_cam_d435** = R_torso→d435^T · R_cam_torso  

  With R_torso→d435 = R_Y(0.8308), so R_torso→d435^T = R_Y(−0.8308):

  - **t_cam** = R_Y(−0.8308) · (p_oak_torso − p_torso_d435)  
  - **q_cam_d435** = q_Y(−0.8308) ⊗ q_oak_torso (quaternion multiplication).

**Numerical result (Camera in d435_link frame):**

- **translate (m):** (0.033976, −0.026905, 0.029194)  
  - Formula: `t_cam = R_Y(−0.8308) @ (p_oak_torso − p_torso_d435)` with p_oak_torso and p_torso_d435 as above.
- **orient (USD quatd: x, y, z, w):** (−0.5077606323, −0.510929544, 0.4795505526, 0.5011600631)  
  - Raw match then 180° roll (orient ⊗ q_z180) so the rendered image is right-side up, not upside down.

The **raw** computed orient makes the combined pose match MuJoCo but the **rendered image is upside down** (MuJoCo vs USD/Isaac image-Y convention). Apply a **180° roll** so the image is right-side up: orient_final = orient_raw ⊗ q_z180, q_z180 = (0,0,1,0) in (x,y,z,w) → (x,y,z,w)_final = (oy, -ox, ow, -oz).

So in the USD, under `def Camera "Camera"` inside `d435_link`, set:

```usda
double3 xformOp:translate = (0.033976, -0.026905, 0.029194)
quatd xformOp:orient = (-0.5077606323, -0.510929544, 0.4795505526, 0.5011600631)
```

This keeps position and view direction aligned with MuJoCo and corrects the image to be right-side up.

**Applying the USD patch:** Run (from sim2behavior, with OmniGibson/Isaac env so `pxr` is available):

```bash
python scripts/patch_g1_usd_d435_oak_egoview.py [path/to/unitree_g1.usda]
```

If no path is given, the script uses `OMNIGIBSON_DATA_PATH` or `BEHAVIOR_DATA_PATH` + `omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda`. It backs up the USD before modifying.

### 6.4 Matching FOV and other parameters

- **MuJoCo oak_egoview:** **fovy = 79.5°** (vertical field of view), resolution 640×480.
- **OmniGibson / USD:** FOV is determined by **focal length** and **aperture**. Relation:  
  **fovy_rad = 2 · atan( (verticalAperture/2) / focalLength )**  
  So for a desired fovy (e.g. 79.5°):  
  **focalLength = (verticalAperture/2) / tan(fovy_rad/2)**.

  With 640×480, aspect = 480/640. If **horizontalAperture = 12** mm (common in USD), **verticalAperture = 12 × (480/640) = 9** mm. Then:
  - fovy_rad = 79.5° × π/180 ≈ 1.387 rad  
  - **focalLength ≈ (9/2) / tan(39.75°) ≈ 5.41 mm**

  So to match MuJoCo’s 79.5° vertical FOV for the ego camera:
  - **In the USD:** You can set the Camera prim’s **focalLength** (e.g. to **5.41** mm) and **horizontalAperture** (e.g. **12** mm) if the pipeline reads them. Isaac/OmniGibson often use **sensor_kwargs.focal_length** from the **robot config** for VisionSensors.
  - **In the config (recommended):** Under `sensor_config.VisionSensor.sensor_kwargs`, set **focal_length: 5.41** for the robot so that the D435 (and any other onboard camera using the same config) gets ~79.5° vertical FOV.  
  - **Note:** `sensor_config.VisionSensor` applies to **all** robot VisionSensors in the current API. If you need 79.5° only for the D435 and a different FOV for mid360, you would need a per-camera mechanism or a separate config path; otherwise use 5.41 for full MuJoCo match and accept the same FOV for all onboard cams.

**Summary of parameters to match:**

| Parameter        | MuJoCo oak_egoview | OmniGibson / USD (D435) |
|------------------|--------------------|---------------------------|
| Resolution       | 640×480            | `image_width: 640`, `image_height: 480` in config |
| Vertical FOV     | 79.5°              | `focal_length: 5.41` (mm) in sensor_kwargs (with horizontalAperture 12 mm) |
| Pose in torso    | pos, quat above    | USD: translate (0.033976, -0.026905, 0.029194), orient (−0.5078..., −0.5109..., 0.4796..., 0.5012...) in d435_link frame (with 180° roll so image right-side up) |
