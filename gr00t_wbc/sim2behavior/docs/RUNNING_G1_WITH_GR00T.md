# Running G1 Robot with GR00T in Behavior Environment

This document summarizes the command lines for running the Unitree G1 robot driven by the GR00T policy in OmniGibson, MuJoCo, and BEHAVIOR environments.

---

## Option 1: OmniGibson + G1 + GR00T (sim2behavior)

This runs the G1 in OmniGibson with the GR00T policy via observation and action adapters.

### Terminal 1 – GR00T policy server

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper
```

### Terminal 2 – OmniGibson client

Activate your BEHAVIOR/OmniGibson environment first (e.g. `conda activate behavior`), then run from the **Isaac-GR00T** repo root or from `gr00t_wbc/sim2behavior`:

```bash
python scripts/run_omnigibson_g1_with_gr00t.py \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --task_description "pick up the apple and place it on the plate" \
    --save_video \
    --config_path configs/g1_with_scene.yaml
```

Or from Isaac-GR00T repo root:

```bash
python external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2behavior/scripts/run_omnigibson_g1_with_gr00t.py \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --task_description "pick up the apple and place it on the plate"
```

### With scene geometry and video

```bash
python scripts/run_omnigibson_g1_with_gr00t.py \
    --config_path configs/g1_with_scene.yaml \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --save_video \
    --skip_close
```

### Topdown camera following the robot

By default the topdown camera is fixed. Use `--topdown_follow_robot` to keep it above the robot. If the video shows sky instead of the robot, try `--topdown_orientation identity`, `flip_x`, `flip_y`, or `flip_z`:

```bash
OMNIGIBSON_SAVE_SCENE_USD_TO=/mnt/nas26/qiang.liu/Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2behavior/output_frames/Rs_int_scene.usd \
OMNIGIBSON_SAVE_SCENE_EXCLUDE_CATEGORIES=ceilings \
python scripts/run_omnigibson_g1_with_gr00t.py \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --save_video \
    --topdown_follow_robot \
    --topdown_follow_height 0.5 \
    --topdown_orientation identity \
    --config_path configs/g1_with_scene.yaml
```

### Local policy (no server)

```bash
python scripts/run_omnigibson_g1_with_gr00t.py \
    --model_path /path/to/checkpoint \
    --task_description "pick up the apple"
```

### Prerequisites

- BEHAVIOR/OmniGibson virtual environment must be activated.
- Set `OMNIGIBSON_DATA_PATH` to your BEHAVIOR-1K dataset root (e.g. `/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets`) if using scenes or robot assets.
- See [SCENE_WITH_GEOMETRY.md](SCENE_WITH_GEOMETRY.md) for scene config requirements.

---

## Option 2: MuJoCo G1 Loco-Manip (GR00T-WholeBodyControl)

Recommended for evaluation with G1 and whole-body control (gear_wbc).

### Terminal 1 – GR00T server

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
    --embodiment-tag UNITREE_G1 \
    --use-sim-policy-wrapper
```

### Terminal 2 – Rollout client

Use the GR00T-WholeBodyControl venv:

```bash
gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 1440 \
    --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
    --n_action_steps 20 \
    --n_envs 5
```

### Available G1 WBC env names

| env_name |
|----------|
| `gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc` |
| `gr00tlocomanip_g1_sim/LMBottlePnP_G1_gear_wbc` |
| `gr00tlocomanip_g1_sim/PnPBottle_G1_gear_wbc` |

### Prerequisites

- One-time setup: `bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh`
- See [examples/GR00T-WholeBodyControl/EVALUATION_G1_WBC.md](../../../../examples/GR00T-WholeBodyControl/EVALUATION_G1_WBC.md) for full evaluation setup.

---

## Option 3: BEHAVIOR 1K Benchmark

Uses the **BEHAVIOR_R1_PRO** embodiment (not G1). For G1, use Option 1 or 2.

### Terminal 1 – Server

```bash
uv run gr00t/eval/run_gr00t_server.py \
    --model-path nvidia/GR00T-N1.6-BEHAVIOR1k \
    --embodiment-tag BEHAVIOR_R1_PRO \
    --use-sim-policy-wrapper
```

### Terminal 2 – Client

```bash
uv run python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps 999999999 \
    --env_name sim_behavior_r1_pro/turning_on_radio \
    --n_action_steps 8 \
    --n_envs 1
```

---

## Quick Reference

| Goal | Embodiment | Client Script |
|------|------------|---------------|
| OmniGibson + G1 + GR00T | `UNITREE_G1` | `run_omnigibson_g1_with_gr00t.py` |
| MuJoCo G1 loco-manip | `UNITREE_G1` | `rollout_policy.py` with `gr00tlocomanip_g1_sim/*` |
| BEHAVIOR 1K benchmark | `BEHAVIOR_R1_PRO` | `rollout_policy.py` with `sim_behavior_r1_pro/*` |

**Server flags for G1:** always use `--embodiment-tag UNITREE_G1` and `--use-sim-policy-wrapper`.
