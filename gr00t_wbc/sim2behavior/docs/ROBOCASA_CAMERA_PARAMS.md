# RoboCasa camera parameters reference

This document summarizes where and how cameras are defined in the **RoboCasa** codebase (MuJoCo/robosuite-based kitchen env). Use it to align sim2behavior/OmniGibson camera settings with RoboCasa when needed.

---

## 1. Source files

| What | File |
|------|------|
| **Extrinsics + intrinsics (FOV)** | `robocasa/utils/camera_utils.py` |
| **Resolution / which cameras are used** | `robocasa/environments/kitchen/kitchen.py` (env init), `robocasa/models/robots/__init__.py` (key converter `get_camera_config`) |
| **Injection into MJCF** | `robocasa/environments/kitchen/kitchen.py` (`edit_model_xml`: sets `pos`, `quat`, and `camera_attribs` on `<camera>` elements) |

---

## 2. Robot observation cameras (`camera_utils.CAM_CONFIGS["DEFAULT"]`)

All poses are in the **parent body frame**. Quaternions are **xyzw**. MuJoCo camera `mode` is **fixed**.

| Camera name | Position `pos` (m) | Quaternion `quat` (xyzw) | Intrinsics | Parent body |
|-------------|-------------------|--------------------------|------------|-------------|
| **robot0_agentview_center** | [-0.6, 0.0, 1.15] | [0.637, 0.333, -0.320, -0.618] | *(no fovy set → MuJoCo default)* | mobilebase0_support |
| **robot0_agentview_left**  | [-0.5, 0.35, 1.05] | [0.556, 0.299, -0.377, -0.678] | **fovy = 60** (°) | mobilebase0_support |
| **robot0_agentview_right** | [-0.5, -0.35, 1.05] | [0.678, 0.377, -0.299, -0.556] | **fovy = 60** (°) | mobilebase0_support |
| **robot0_frontview**       | [-0.50, 0, 0.95] | [0.609, 0.381, -0.367, -0.591] | **fovy = 60** (°) | mobilebase0_support |
| **robot0_eye_in_hand**     | [0.05, 0, 0] | [0, 0.707, 0.707, 0] | *(none)* | robot0_right_hand |

- **PandaMobile** and **GR1FixedLowerBody** use the same config (empty overrides in `CAM_CONFIGS`).

---

## 3. Intrinsics

- **FOV:** Only **vertical FOV** is set where applicable: **`fovy="60"`** (degrees) in `camera_attribs` for agentview_left, agentview_right, frontview. agentview_center and eye_in_hand have no `camera_attribs`, so they use MuJoCo’s default FOV.
- **Focal length:** Not set explicitly; MuJoCo derives it from `fovy` and image size.
- **Resolution:** Set per env and per key converter (see below).

---

## 4. Resolution and camera choice by robot / key converter

From `robocasa/models/robots/__init__.py` (`get_camera_config`):

| Robot / key converter | Camera names (obs) | Width × height |
|----------------------|--------------------|----------------|
| **GR1ArmsOnly** / **GR1ArmsAndWaist** | egoview | 1280 × 800 |
| **GR1FixedLowerBody** | agentview | 1280 × 800 |
| **PandaOmron** | robot0_agentview_left, robot0_agentview_right, robot0_eye_in_hand | 512 × 512 |
| **Panda_Panda** (bimanual) | agentview, robot0_eye_in_hand, robot1_eye_in_hand | 1280 × 800 |
| **PandaDexRH_PandaDexLH** | (same as Panda_Panda) | 1280 × 800 |

Kitchen env defaults (`kitchen.py`): **camera_names="agentview"**, **camera_heights=256**, **camera_widths=256**, **camera_depths=False**. So when using the generic "agentview" name, the rendered obs can be 256×256 unless overridden by the key converter’s `get_camera_config()` (which may request 1280×800 or 512×512).

---

## 5. Layout / viewer cameras (not used for obs)

`camera_utils.LAYOUT_CAMS` and `DEFAULT_LAYOUT_CAM` define **free** viewer cameras (lookat, distance, azimuth, elevation) for different kitchen layouts. Used for rendering the main view (e.g. mjviewer), not for `obs[..._image]`.

---

## 6. Converting to OmniGibson / sim2behavior

- **FOV:** RoboCasa uses **fovy = 60°**. In OmniGibson, approximate with **focal_length** (e.g. for 256×256 and 60° vertical FOV, focal ≈ 256 / (2 * tan(30°)) ≈ 222 px; convert to mm using sensor size if needed), or set a **focal_length** that gives a similar vertical FOV in your resolution.
- **Extrinsics:** RoboCasa poses are in the **parent body** frame (e.g. mobilebase0_support). In OmniGibson you have scene/world frame; convert link frame to world when placing external or robot cameras.
- **Resolution:** Match the key converter (e.g. 1280×800 for GR1, 512×512 for PandaOmron) in `sensor_kwargs.image_width` / `image_height` if you want pixel-aligned behavior.
