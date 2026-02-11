# Using a Scene with Geometry (InteractiveTraversableScene)

When using the minimal config (`g1_standalone.yaml`), the scene type is `Scene` — an empty scene with no floor or walls. The visible "background" is Isaac Sim's default clear color, which can look similar to the robot.

To get a **scene with geometry** (floor, walls, room) so the robot stands out:

## 1. Use the provided scene config

A second config file uses `InteractiveTraversableScene` and a BEHAVIOR scene model:

- **Config:** `configs/g1_with_scene.yaml`
- **Scene type:** `InteractiveTraversableScene`
- **Scene model:** `Rs_int` (simple room; you can change this to any valid BEHAVIOR scene name)

## 2. How to run with the scene config

Pass the config path when starting the script:

```bash
python scripts/run_omnigibson_g1_with_gr00t.py \
  --config_path configs/g1_with_scene.yaml \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  --save_video \
  --skip_close
```

Or for the standalone script (no GR00T):

```bash
# From sim2behavior directory; set CONFIG_PATH or pass --config_path if your script supports it
python scripts/run_omnigibson_g1_standalone.py
# and ensure the script loads configs/g1_with_scene.yaml (e.g. via env or default CONFIG_PATH)
```

## 3. Requirements

- **BEHAVIOR dataset path:** OmniGibson looks for scenes at `{ROOT}/datasets/behavior-1k-assets/scenes`. Set the dataset root before importing OmniGibson:
  - **Option A:** Pass the root when running the script:
    ```bash
    python scripts/run_omnigibson_g1_with_gr00t.py \
      --config_path configs/g1_with_scene.yaml \
      --behavior_data_path /mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets \
      ...
    ```
  - **Option B:** Set an environment variable (before running). Default in script is `/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets`:
    ```bash
    export OMNIGIBSON_DATA_PATH=/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets
    ```
  - Your dataset root must contain either `datasets/behavior-1k-assets` (with a `scenes` folder inside), or the same layout that your OmniGibson version expects. If you get "No such file or directory: .../datasets/behavior-1k-assets/scenes", point `--behavior_data_path` (or `OMNIGIBSON_DATA_PATH`) to the directory that has `datasets/behavior-1k-assets` under it.

- **Scene model name:** In `g1_with_scene.yaml` the default is `scene_model: "Rs_int"`. Override with `--scene_model`:
  ```bash
  python scripts/run_omnigibson_g1_with_gr00t.py --config_path configs/g1_with_scene.yaml --scene_model Beechwood_0_int ...
  ```
  Other valid names include:
  - `Rs_int`, `Rs_garden`
  - `Beechwood_0_int`, `Benevolence_0_int`, `Merom_0_int`, `Pomaria_0_int`, `Wainscott_0_int`
  - `house_single_floor`, `office_large`, `restaurant_diner`, `school_gym`, etc.  
  Full list: [OmniGibson Scenes](https://behavior.stanford.edu/omnigibson/scenes.html).

## 4. Optional scene parameters

In `configs/g1_with_scene.yaml` you can add or adjust:

| Parameter                 | Description                          | Example   |
|--------------------------|--------------------------------------|-----------|
| `scene_model`            | BEHAVIOR scene name                  | `Rs_int`  |
| `trav_map_resolution`    | Traversability map resolution (m)    | `0.1`     |
| `default_erosion_radius` | Erosion radius for traversability    | `0.0`     |
| `trav_map_with_objects`  | Include objects in traversability    | `true`    |
| `num_waypoints`          | Number of waypoints                  | `10`      |
| `waypoint_resolution`    | Waypoint spacing                     | `0.2`     |

If the scene fails to load, check that `--behavior_data_path` or `OMNIGIBSON_DATA_PATH` points to the dataset root that contains `datasets/behavior-1k-assets/scenes` (or the layout required by your OmniGibson version), and that the requested `scene_model` exists there.

### Saving the scene USD

To save the prebuilt scene USD to a file (e.g. for inspection or reuse), set the environment variable before running:

```bash
# Rs_int (default)
OMNIGIBSON_SAVE_SCENE_USD_TO=output_frames/Rs_int_scene.usd python scripts/run_omnigibson_g1_with_gr00t.py --config_path configs/g1_with_scene.yaml ...

# Another scene (e.g. Beechwood_0_int)
OMNIGIBSON_SAVE_SCENE_USD_TO=output_frames/Beechwood_0_int_scene.usd python scripts/run_omnigibson_g1_with_gr00t.py --config_path configs/g1_with_scene.yaml --scene_model Beechwood_0_int ...
```

The file is written when the scene is first prebuilt (before the first simulation step).

To save a scene USD **without ceilings** (so a top-down snapshot is not blocked), set `OMNIGIBSON_SAVE_SCENE_EXCLUDE_CATEGORIES=ceilings` when exporting. Example:

```bash
OMNIGIBSON_SAVE_SCENE_USD_TO=output_frames/Rs_int_scene.usd OMNIGIBSON_SAVE_SCENE_EXCLUDE_CATEGORIES=ceilings python scripts/run_omnigibson_g1_with_gr00t.py --config_path configs/g1_with_scene.yaml ...
```

The simulation still loads the full scene; only the saved file excludes those categories.

### Visualizing the saved scene USD

To render a snapshot of a saved scene USD (headless, RTX disabled) and save as PNG:

```bash
# Rs_int
python scripts/visualize_scene_usd.py output_frames/Rs_int_scene.usd -o output_frames/Rs_int_snapshot.png

# Another scene (after saving it first)
python scripts/visualize_scene_usd.py output_frames/Beechwood_0_int_scene.usd -o output_frames/Beechwood_0_int_snapshot.png
```

Uses the same OmniGibson/Isaac Sim Python environment as the main scripts.

## Troubleshooting: empty scene, layout mismatch, no room labels

### 1. Scene USD has only a few objects (e.g. 3–4), looks almost empty

The file written by `OMNIGIBSON_SAVE_SCENE_USD_TO` is a **copy of the prebuilt scene USD**: it contains exactly the objects that were included when the scene was **prebuilt** (at first load). That set is determined by:

- The scene JSON (e.g. `Rs_int_best.json`) and
- OmniGibson’s **`load_object_categories`** / **`load_room_instances`** (and related) filters.

If you only see floors and walls (or very few objects):

- **Do not** set `load_object_categories` to a short list (e.g. `["floors", "walls"]`). In `configs/g1_with_scene.yaml`, leave **`load_object_categories` unset** (or omit it) so that **all** object categories from the scene file are loaded.
- Re-run the **same** run that creates the USD (with `OMNIGIBSON_SAVE_SCENE_USD_TO` set) using that config, then re-run `visualize_scene_usd.py` on the new file. The prebuild runs once per run; the saved USD is a snapshot of that prebuild.
- If you use a different config or script that restricts categories (e.g. structure-only for speed), the exported USD will only contain those categories.

### 2. Walls / layout don’t match the 2D floor plan (e.g. `floor_trav_0.png`) — e.g. “4 equal rooms” instead of living room 4× bathroom

**Cause:** The file written by `OMNIGIBSON_SAVE_SCENE_USD_TO` is a copy of OmniGibson’s **prebuilt** scene USD. Previously, the prebuild only added **asset references** (walls, floors, furniture) and did **not** write each object’s **position and orientation** from the scene JSON. So in the saved USD every object sat at the default pose (e.g. origin), which made the layout look wrong (e.g. four same-sized quadrants instead of the real floor plan where the living room is much larger than the bathroom).

**Fix (OmniGibson):** In BEHAVIOR-1K’s OmniGibson, `scene_base.py` was updated so that when the scene is prebuilt, each object’s **transform** (position and orientation from the scene JSON `_init_state`) is applied to the prim before saving. After that change, the saved USD matches the floor plan and the 2D layout image.

- **Re-export the scene USD** after pulling the OmniGibson change: run your script again with `OMNIGIBSON_SAVE_SCENE_USD_TO=...` set, then re-run `visualize_scene_usd.py` on the new file. The snapshot should now match `floor_trav_0.png` proportions (living room largest, bathroom smallest, etc.).
- Use **`--up-axis z`** in `visualize_scene_usd.py` when the scene is Z-up so the camera looks straight down the Z-axis at the X–Y floor.
- The 2D layout images under `scenes/<scene_model>/layout/` are top-down maps; with the fix, the 3D view and 2D layout should agree.

### 3. Can’t tell which room is kitchen, bedroom, living room, etc.

The saved **scene USD contains only geometry** (meshes, transforms). It does **not** contain room semantics (kitchen, bedroom, etc.). Room labels come from BEHAVIOR’s **segmentation map** (e.g. `floor_semseg_0.png`) and OmniGibson’s `SegmentationMap` / scene metadata, which are separate from the USD.

- To see which region is which room, use the layout assets (e.g. `layout/floor_semseg_0.png` or the Knowledgebase) or query the scene in OmniGibson (e.g. `seg_map`) when the full environment is loaded.
- The standalone `visualize_scene_usd.py` script does not load segmentation data; it only renders the USD.

## Where does `download_behavior_1k_assets()` put the data?

`download_behavior_1k_assets(accept_license=True)` uses OmniGibson’s `get_dataset_path("behavior-1k-assets")`. That path is derived from OmniGibson’s **data root** (the same root used for `omnigibson-robot-assets` and scenes). In practice:

- The assets are written under **`{DATA_ROOT}/datasets/behavior-1k-assets`** (with a `scenes` folder inside).
- **DATA_ROOT** is set when OmniGibson is first imported. It usually comes from the **`OMNIGIBSON_DATA_PATH`** environment variable if that is set; otherwise OmniGibson uses its built-in default (e.g. under your home directory or an install-specific path).

So:

1. **To choose the install location:** set `OMNIGIBSON_DATA_PATH` to your desired dataset root **before** running the download (and before any `import omnigibson`):

   ```bash
   export OMNIGIBSON_DATA_PATH=/path/to/your/BEHAVIOR-1K-datasets
   python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"
   ```

   The files will then appear under `/path/to/your/BEHAVIOR-1K-datasets/datasets/behavior-1k-assets/` (including `scenes/`).

2. **To see where it would go** (without downloading), set `OMNIGIBSON_DATA_PATH` if desired, then run:

   ```bash
   export OMNIGIBSON_DATA_PATH=/path/to/your/root   # optional
   python -c "from omnigibson.utils.asset_utils import get_dataset_path; print(get_dataset_path('behavior-1k-assets'))"
   ```
