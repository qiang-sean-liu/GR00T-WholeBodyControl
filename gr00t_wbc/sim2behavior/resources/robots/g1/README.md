# Unitree G1 for sim2behavior (OmniGibson/BEHAVIOR)

- **urdf/g1_29dof_with_hand.urdf** – OmniGibson-compatible URDF (world + floating base; from sim2mujoco, MuJoCo block removed).
- **g1_omnigibson_config.yaml** – Robot path, controllers (base, waist, arms, Dex3_1 hands), and GR00T obs/action mapping. The standalone script uses it only for `robot.urdf_path` and `robot.meshes_dir`. **Runtime controller behavior** is defined in **configs/g1_standalone.yaml** (passed to OmniGibson); keep the two in sync.
- **meshes/** – Symlink to `sim2mujoco/resources/robots/g1/meshes`. Create with:
  ```bash
  python ../../../../scripts/setup_meshes_symlink.py
  ```
  Or from this directory:
  ```bash
  ln -snf ../../../../sim2mujoco/resources/robots/g1/meshes meshes
  ```

## How to run

From the **sim2behavior** root (or from the Isaac-GR00T repo with the path below), activate your BEHAVIOR/OmniGibson venv, then:

```bash
# From gr00t_wbc/sim2behavior
python3 scripts/run_omnigibson_g1_standalone.py
```

Or from the Isaac-GR00T repo root:

```bash
python3 external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2behavior/scripts/run_omnigibson_g1_standalone.py
```

The script checks the config and G1 assets (URDF + meshes). If OmniGibson is not installed, it exits after validation. With OmniGibson and an optional `configs/g1_standalone.yaml` scene config, it launches a minimal sim and runs a step loop. See the main [sim2behavior README](../../README.md) and [sim2behavior_DESIGN.md](../../../sim2behavior_DESIGN.md) §11 for venv setup.
