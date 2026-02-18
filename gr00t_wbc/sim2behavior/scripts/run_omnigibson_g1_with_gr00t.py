#!/usr/bin/env python3
"""
Run OmniGibson G1 robot with GR00T policy (using observation and action adapters).

This script:
1. Creates an OmniGibson environment with the G1 robot
2. Connects to a GR00T policy server (or loads policy locally)
3. Each step: gets OmniGibson obs → adapts to GR00T format → gets action from policy → adapts to OmniGibson → env.step()

Usage:
    # Terminal 1: Start GR00T server
    uv run python gr00t/eval/run_gr00t_server.py \
        --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
        --embodiment-tag UNITREE_G1 \
        --use-sim-policy-wrapper

    # Terminal 2: Run this script (OmniGibson client)
    python scripts/run_omnigibson_g1_with_gr00t.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --task_description "pick up the apple and place it on the plate"

Or with local policy (no server):
    python scripts/run_omnigibson_g1_with_gr00t.py \
        --model_path /path/to/checkpoint \
        --task_description "pick up the apple"

With topdown camera following the robot:
    python scripts/run_omnigibson_g1_with_gr00t.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --save_video \
        --topdown_follow_robot \
        --topdown_follow_height 2.7

With Whole-Body Control (WBC) for lower-body locomotion, same as RoboCasa locomanip:
    python scripts/run_omnigibson_g1_with_gr00t.py \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --use_wbc

Dump per-step actions (raw GR00T, WBC goal, WBC output) for debugging:
    python scripts/run_omnigibson_g1_with_gr00t.py ... --dump_actions [--dump_actions_dir /path]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import warnings

import numpy as np

# Suppress noisy Gymnasium warning
warnings.filterwarnings("ignore", message="Casting input x to numpy array", module="gymnasium.spaces.box")

# Headless + disable RTX/XR (before any omnigibson import)
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("ISAAC_SIM_HEADLESS", "1")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("RTX_DRIVER_VERIFICATION", "0")
# Isaac Sim arguments (these need to be in sys.argv for OmniGibson but not for argparse)
_isaac_args = []
# _isaac_args = [
#     "--/rtx/verifyDriverVersion/enabled=false",
#     "--/rtx/enabled=false",
#     "--/app/extensions/omni.kit.xr.core/disabled=true",
#     "--/app/extensions/omni.kit.xr.profile.vr/disabled=true",
#     "--/app/extensions/omni.kit.xr/disabled=true",
# ]
# Filter out Isaac Sim args from sys.argv before argparse sees them
_isaac_args_in_argv = []
_filtered_argv = []
for arg in sys.argv:
    if arg in _isaac_args:
        _isaac_args_in_argv.append(arg)
    else:
        _filtered_argv.append(arg)
# Temporarily replace sys.argv for argparse
_original_argv = sys.argv
sys.argv = _filtered_argv

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM2BEHAVIOR_ROOT = os.path.dirname(SCRIPT_DIR)
G1_RESOURCES = os.path.join(SIM2BEHAVIOR_ROOT, "resources", "robots", "g1")
CONFIG_PATH = os.path.join(SIM2BEHAVIOR_ROOT, "configs", "g1_standalone.yaml")

# Default BEHAVIOR dataset root (robot assets + scene assets). OmniGibson looks for
# omnigibson-robot-assets and behavior-1k-assets (or datasets/behavior-1k-assets) under this path.
# Set OMNIGIBSON_DATA_PATH or use --behavior_data_path to override.
_DEFAULT_BEHAVIOR_DATA_PATH = "/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets"

# Add Isaac-GR00T repo root to Python path (for gr00t imports)
# Script is at: Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/sim2behavior/scripts/...
# Repo root is: Isaac-GR00T/ (5 levels up from script)
ISAAC_GR00T_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../../../"))
if ISAAC_GR00T_ROOT not in sys.path:
    sys.path.insert(0, ISAAC_GR00T_ROOT)

# Add GR00T-WholeBodyControl root to Python path (for gr00t_wbc imports)
# gr00t_wbc is at: Isaac-GR00T/external_dependencies/GR00T-WholeBodyControl/gr00t_wbc/
GR00T_WBC_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if GR00T_WBC_ROOT not in sys.path:
    sys.path.insert(0, GR00T_WBC_ROOT)


def _tensor_to_python(obj):
    """Convert torch/numpy to native Python for YAML serialization."""
    if obj is None:
        return None
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _tensor_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tensor_to_python(x) for x in obj]
    return obj


def _format_number_6(x):
    """Format a single number as a 6-character string (e.g. -0.650374 -> '-0.650', 1 -> '     1')."""
    if isinstance(x, bool):
        return "  True" if x else " False"
    if isinstance(x, int):
        s = f"{x:6d}"
        return s[-6:] if len(s) > 6 else s.rjust(6)
    if isinstance(x, (float, np.floating)):
        s = f"{float(x):.4f}"
        s = s[:6] if len(s) >= 6 else s.rjust(6)
        return s
    return str(x)[:6].rjust(6)


def _format_dump_6(obj):
    """Recursively format all numbers in obj (dict/list) to 6-char strings for action dump."""
    if isinstance(obj, dict):
        return {k: _format_dump_6(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_format_dump_6(x) for x in obj]
    if isinstance(obj, (int, float, np.integer, np.floating, bool)):
        return _format_number_6(obj)
    return obj


def _dump_camera_params(og, out_dir: str) -> None:
    """Write intrinsics and extrinsics for all VisionSensors to out_dir/camera_params.yaml."""
    try:
        VisionSensor = og.sensors.VisionSensor
    except AttributeError:
        try:
            from omnigibson.sensors import VisionSensor
        except ImportError:
            print("DUMP_CAMERAS=1: could not import VisionSensor", file=sys.stderr)
            return
    out_path = os.path.join(out_dir, "camera_params.yaml")
    report = {}
    for prim_path, sensor in list(getattr(VisionSensor, "SENSORS", {}).items()):
        if not getattr(sensor, "initialized", False):
            continue
        name = sensor.name if hasattr(sensor, "name") else prim_path.split("/")[-1] or prim_path
        entry = {"prim_path": prim_path}
        try:
            params = sensor.camera_parameters
            entry["focal_length_mm"] = float(params.get("cameraFocalLength", 0))
            entry["clipping_range"] = _tensor_to_python(params.get("cameraNearFar"))
            entry["aperture"] = _tensor_to_python(params.get("cameraAperture"))
            view = params.get("cameraViewTransform")
            if view is not None:
                entry["camera_view_transform_world_from_camera_4x4"] = _tensor_to_python(view)
        except Exception as e:
            entry["error"] = str(e)
        try:
            entry["intrinsic_matrix_3x3"] = _tensor_to_python(sensor.intrinsic_matrix)
        except Exception as e:
            entry["intrinsic_error"] = str(e)
        report[name] = entry
    if not report:
        print("DUMP_CAMERAS=1: no initialized VisionSensors found", file=sys.stderr)
        return
    try:
        import yaml
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.safe_dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Dumped camera params: {out_path}")
    except Exception as e:
        print(f"DUMP_CAMERAS=1: could not write {out_path}: {e}", file=sys.stderr)


def _get_camera_extrinsics_per_frame(og, camera_names=None):
    """
    Extract camera extrinsics (view transform) for specific cameras at current frame.
    
    Args:
        og: omnigibson module
        camera_names: List of camera names to extract (e.g., ["d435", "topdown"]).
                     If None, extracts all cameras.
    
    Returns:
        dict: {camera_name: {position: [x,y,z], view_transform_world_from_camera_4x4: [...], ...}}
    """
    try:
        VisionSensor = og.sensors.VisionSensor
    except AttributeError:
        try:
            from omnigibson.sensors import VisionSensor
        except ImportError:
            return {}
    
    result = {}
    for prim_path, sensor in list(getattr(VisionSensor, "SENSORS", {}).items()):
        if not getattr(sensor, "initialized", False):
            continue
        name = sensor.name if hasattr(sensor, "name") else prim_path.split("/")[-1] or prim_path
        
        # Filter by camera_names if provided
        if camera_names is not None:
            if not any(cam_name.lower() in name.lower() for cam_name in camera_names):
                continue
        
        entry = {}
        try:
            params = sensor.camera_parameters
            view = params.get("cameraViewTransform")
            if view is not None:
                view_matrix_raw = _tensor_to_python(view)
                entry["view_transform_world_from_camera_4x4"] = view_matrix_raw
                
                # Extract position from view transform (4x4 matrix, column-major list of 16 elements)
                # Matrix is stored column-major: [R00, R10, R20, 0, R01, R11, R21, 0, R02, R12, R22, 0, Tx, Ty, Tz, 1]
                # Position (translation) is at indices 12, 13, 14
                if isinstance(view_matrix_raw, list) and len(view_matrix_raw) == 16:
                    # Position is at indices 12, 13, 14 (translation column)
                    entry["position"] = [
                        float(view_matrix_raw[12]),
                        float(view_matrix_raw[13]),
                        float(view_matrix_raw[14])
                    ]
                else:
                    # Try numpy if available
                    try:
                        matrix = np.array(view_matrix_raw)
                        if matrix.shape == (4, 4):
                            entry["position"] = matrix[:3, 3].tolist()
                    except Exception:
                        pass
        except Exception as e:
            entry["error"] = str(e)
        
        if entry:
            result[name] = entry
    
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OmniGibson G1 with GR00T policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy_client_host",
        type=str,
        default="",
        help="GR00T policy server host (empty = use local model_path)",
    )
    parser.add_argument(
        "--policy_client_port",
        type=int,
        default=5555,
        help="GR00T policy server port",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Local GR00T checkpoint path (if not using server)",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default="pick up the object",
        help="Language instruction for the task",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="Maximum simulation steps",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save video frames to output_frames/",
    )
    parser.add_argument(
        "--dump_camera_extrinsics_per_frame",
        action="store_true",
        help="Dump D435 and topdown camera extrinsics for each frame to output_frames/camera_extrinsics_per_frame.json",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=30,
        help="Frames per second for saved video (default: 1)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=CONFIG_PATH,
        help="Path to OmniGibson config YAML",
    )
    parser.add_argument(
        "--scene_model",
        type=str,
        default="",
        help="Override scene_model in config (e.g. Beechwood_0_int, house_single_floor). Only applies when config has a scene.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information about observations and actions",
    )
    parser.add_argument(
        "--skip_close",
        action="store_true",
        help="Skip env.close() to avoid segfault (same as OMNIGIBSON_SKIP_CLOSE=1)",
    )
    parser.add_argument(
        "--behavior_data_path",
        type=str,
        default="",
        help="BEHAVIOR dataset root (parent of omnigibson-robot-assets and, if present, datasets/behavior-1k-assets). Default: OMNIGIBSON_DATA_PATH or a built-in path.",
    )
    parser.add_argument(
        "--topdown_follow_robot",
        action="store_true",
        help="Update the topdown external camera to follow the robot each step (camera stays above robot).",
    )
    parser.add_argument(
        "--topdown_follow_height",
        type=float,
        default=0.5,
        help="Height offset (m) above the robot when --topdown_follow_robot is used (default: 10).",
    )
    parser.add_argument(
        "--topdown_orientation",
        type=str,
        default="identity",
        choices=["identity", "flip_x", "flip_y", "flip_z"],
        help="Topdown camera orientation (identity=look down in Isaac; try flip_x/flip_y/flip_z if view is wrong).",
    )
    parser.add_argument(
        "--use_wbc",
        action="store_true",
        help="Use Whole-Body Control (WBC) to compute lower-body targets from GR00T navigate/base_height + upper-body; same pipeline as RoboCasa locomanip.",
    )
    parser.add_argument(
        "--dump_actions",
        action="store_true",
        help="Dump per-step actions to disk: raw GR00T output, WBC goal (navigate_cmd, base_height, target_upper_body_pose), and WBC output (q).",
    )
    parser.add_argument(
        "--dump_actions_dir",
        type=str,
        default="",
        help="Directory for action dumps (default: <frames_dir>/action_dumps or output_frames/action_dumps).",
    )
    parser.add_argument(
        "--hold_initial_pose",
        action="store_true",
        help="Send the initial standing pose (from step-0 observation) as action every step so the robot stands still.",
    )
    parser.add_argument(
        "--inject_standing_reset",
        action="store_true",
        help="Inject robot_model.default_body_pose as reset_joint_pos so the robot starts in a standing pose (fixes jump/collapse when no reset_joint_pos in YAML).",
    )
    parser.add_argument(
        "--no_op_first_step",
        action="store_true",
        help="On step 0 only, send current pose as action (no-op) so the first transition never uses the policy; avoids snap from policy's first-frame command.",
    )
    parser.add_argument(
        "--clamp_sent_indices_4_10",
        type=float,
        default=None,
        metavar="VALUE",
        help="Overwrite sent_to_simulator indices 4 and 10 (base DOFs) with this value, e.g. 0.1, to test if they cause snap off floor.",
    )
    parser.add_argument(
        "--clamp_sent_index_14",
        type=float,
        default=None,
        metavar="VALUE",
        help="Overwrite sent_to_simulator index 14 (base DOF) with this value, e.g. 0.01.",
    )
    parser.add_argument(
        "--zero_base_action",
        action="store_true",
        help="Set the first 15 sent_to_simulator values (base/lower-body) to 0. Use with base controller use_delta_commands: true so base holds current pose.",
    )
    parser.add_argument(
        "--enable_full_robot_gravity",
        action="store_true",
        help="Re-enable gravity on all robot links (overrides OmniGibson ControllableObject default that disables gravity on non-base links). Use so the robot falls under gravity instead of floating.",
    )
    args = parser.parse_args()

    # Set OMNIGIBSON_SKIP_CLOSE if --skip_close flag is used
    if args.skip_close:
        os.environ["OMNIGIBSON_SKIP_CLOSE"] = "1"

    # Set BEHAVIOR/OmniGibson dataset path before any omnigibson import (for robot USD and scene assets)
    behavior_data_path = args.behavior_data_path or os.environ.get("OMNIGIBSON_DATA_PATH", "")
    if not behavior_data_path:
        behavior_data_path = _DEFAULT_BEHAVIOR_DATA_PATH
    if os.path.isdir(behavior_data_path):
        os.environ["OMNIGIBSON_DATA_PATH"] = behavior_data_path
        # Some OmniGibson/BEHAVIOR code resolves scenes from a root that contains datasets/behavior-1k-assets
        if "BEHAVIOR_DATA_PATH" in os.environ:
            pass  # already set
        else:
            os.environ["BEHAVIOR_DATA_PATH"] = behavior_data_path
        if args.verbose:
            print(f"Using BEHAVIOR/OmniGibson data path: {behavior_data_path}")
    else:
        print(f"Warning: behavior_data_path not found: {behavior_data_path}", file=sys.stderr)
        print("Set --behavior_data_path or OMNIGIBSON_DATA_PATH to the BEHAVIOR-1K dataset root.", file=sys.stderr)

    # Restore sys.argv with Isaac Sim arguments (needed for OmniGibson initialization)
    sys.argv = _original_argv
    # Ensure Isaac Sim args are present
    for arg in _isaac_args:
        if arg not in sys.argv:
            sys.argv.append(arg)

    # Validate: must have either server or model_path
    if not args.policy_client_host and not args.model_path:
        print("Error: Must provide either --policy_client_host or --model_path", file=sys.stderr)
        return 1

    try:
        import omnigibson as og
    except ImportError:
        print("OmniGibson not installed. Install per https://behavior.stanford.edu/omnigibson/", file=sys.stderr)
        return 1

    # Import GR00T policy and adapters (conditional based on mode)
    # Note: gr00t is in the Isaac-GR00T repo root, which should already be in sys.path
    if args.policy_client_host:
        # Server mode: only need PolicyClient
        try:
            from gr00t.policy.server_client import PolicyClient
        except ImportError as e:
            error_msg = str(e)
            print(f"GR00T policy client not available: {error_msg}", file=sys.stderr)
            if "msgpack" in error_msg.lower():
                print("\nMissing dependency: msgpack", file=sys.stderr)
                print("Install it with: pip install msgpack", file=sys.stderr)
            elif "zmq" in error_msg.lower():
                print("\nMissing dependency: pyzmq (ZMQ bindings for Python)", file=sys.stderr)
                print("Install it with: pip install pyzmq", file=sys.stderr)
            elif "gr00t" in error_msg.lower():
                print(f"Make sure Isaac-GR00T repo root ({ISAAC_GR00T_ROOT}) is accessible.", file=sys.stderr)
                print("The gr00t package should be in the repo root (Isaac-GR00T/gr00t/).", file=sys.stderr)
            else:
                print("This might be a missing dependency. Check gr00t requirements.", file=sys.stderr)
            return 1
        Gr00tPolicy = None
        Gr00tSimPolicyWrapper = None
        EmbodimentTag = None
    else:
        # Local model mode: need full policy classes
        try:
            from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
            from gr00t.policy.server_client import PolicyClient
            from gr00t.data.embodiment_tags import EmbodimentTag
        except ImportError as e:
            error_msg = str(e)
            print(f"GR00T policy not available: {error_msg}", file=sys.stderr)
            if "msgpack" in error_msg.lower():
                print("\nMissing dependency: msgpack", file=sys.stderr)
                print("Install it with: pip install msgpack", file=sys.stderr)
            elif "zmq" in error_msg.lower():
                print("\nMissing dependency: pyzmq (ZMQ bindings for Python)", file=sys.stderr)
                print("Install it with: pip install pyzmq", file=sys.stderr)
            elif "gr00t" in error_msg.lower():
                print(f"Make sure Isaac-GR00T repo root ({ISAAC_GR00T_ROOT}) is accessible.", file=sys.stderr)
                print("The gr00t package should be in the repo root (Isaac-GR00T/gr00t/).", file=sys.stderr)
            else:
                print("This might be a missing dependency. Check gr00t requirements.", file=sys.stderr)
            print("Or use --policy_client_host to connect to a server instead.", file=sys.stderr)
            return 1

    try:
        from gr00t_wbc.control.robot_model.instantiation import get_robot_type_and_model
        # Import adapters from the same directory
        sys.path.insert(0, SCRIPT_DIR)
        from omnigibson_gr00t_adapters import (
            OmniGibsonToGR00TObservationAdapter,
            GR00TToOmniGibsonActionAdapter,
        )
    except ImportError as e:
        error_msg = str(e)
        print(f"GR00T-WholeBodyControl adapters not available: {error_msg}", file=sys.stderr)
        if "pinocchio" in error_msg.lower():
            print("\nMissing dependency: pinocchio (robotics library)", file=sys.stderr)
            print("Install it with: pip install pin", file=sys.stderr)
            print("Or: conda install -c conda-forge pinocchio", file=sys.stderr)
        elif "gr00t_wbc" in error_msg.lower():
            print(f"Make sure gr00t_wbc is accessible. GR00T_WBC_ROOT: {GR00T_WBC_ROOT}", file=sys.stderr)
            print(f"Expected location: {os.path.join(GR00T_WBC_ROOT, 'gr00t_wbc')}", file=sys.stderr)
        else:
            print("This might be a missing dependency. Check gr00t_wbc requirements.", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Load OmniGibson config
    import yaml
    if not os.path.isfile(args.config_path):
        print(f"Config not found: {args.config_path}", file=sys.stderr)
        return 1
    with open(args.config_path) as f:
        env_config = yaml.safe_load(f)

    # Override scene_model if specified
    if args.scene_model and "scene" in env_config:
        env_config["scene"] = dict(env_config["scene"])
        env_config["scene"]["scene_model"] = args.scene_model
        print(f"Using scene_model: {args.scene_model}")

    # Create OmniGibson environment
    print("Creating OmniGibson environment...")
    try:
        env = og.Environment(configs=env_config)
    except Exception as e:
        print(f"Failed to create OmniGibson environment: {e}", file=sys.stderr)
        return 1

    # Get robot model for adapters
    robot_name_in_config = env_config.get("robots", [{}])[0].get("name", "unitree_g1")
    print(f"Instantiating robot model for '{robot_name_in_config}'...")
    # get_robot_type_and_model expects robot name starting with "g1" (case-insensitive)
    _, robot_model = get_robot_type_and_model("G1", enable_waist_ik=False)

    # Create policy (server or local) - need to create before adapters to get time horizon
    video_time_horizon = 1  # Default
    state_time_horizon = 1  # Default
    if args.policy_client_host:
        print(f"Connecting to GR00T policy server at {args.policy_client_host}:{args.policy_client_port}...")
        policy = PolicyClient(host=args.policy_client_host, port=args.policy_client_port)
        # For server mode, can't easily get config, use default T=1
    else:
        print(f"Loading GR00T policy from {args.model_path}...")
        policy_base = Gr00tPolicy.from_pretrained(args.model_path, embodiment_tag=EmbodimentTag.UNITREE_G1)
        policy_wrapper = Gr00tSimPolicyWrapper(policy_base)
        # Try to get time horizons from policy modality configs
        try:
            modality_configs = policy_wrapper.get_modality_config()
            if "video" in modality_configs and hasattr(modality_configs["video"], "delta_indices"):
                video_time_horizon = len(modality_configs["video"].delta_indices)
                print(f"Detected video time horizon: T={video_time_horizon}")
            if "state" in modality_configs and hasattr(modality_configs["state"], "delta_indices"):
                state_time_horizon = len(modality_configs["state"].delta_indices)
                print(f"Detected state time horizon: T={state_time_horizon}")
        except Exception as e:
            print(f"Could not determine time horizons, using defaults (video T=1, state T=1): {e}")
        policy = policy_wrapper
    policy.reset()

    # Create adapters (with correct time horizons)
    obs_adapter = OmniGibsonToGR00TObservationAdapter(
        robot_model=robot_model,
        robot_name=robot_name_in_config,
        task_description=args.task_description,
        video_time_horizon=video_time_horizon,
        state_time_horizon=state_time_horizon,
    )
    action_adapter = GR00TToOmniGibsonActionAdapter(
        robot_model=robot_model,
        robot_name=robot_name_in_config,
    )
    
    # Get OmniGibson action space info to map joints correctly
    # OmniGibson may have fewer controllable joints than GR00T's full joint set
    og_action_dim = None
    og_joint_names = None
    if hasattr(env, 'robots') and len(env.robots) > 0:
        robot = env.robots[0]
        if hasattr(robot, 'action_dim'):
            og_action_dim = robot.action_dim
        if hasattr(robot, 'joint_names'):
            og_joint_names = robot.joint_names
        elif hasattr(robot, 'controllable_joints'):
            og_joint_names = [j.name for j in robot.controllable_joints]
    
    # Check action_space shape
    import gymnasium as gym
    if isinstance(env.action_space, gym.spaces.Dict):
        if robot_name_in_config in env.action_space.spaces:
            og_action_dim = env.action_space.spaces[robot_name_in_config].shape[0]
    elif isinstance(env.action_space, gym.spaces.Box):
        og_action_dim = env.action_space.shape[0]
    
    if args.verbose:
        print(f"OmniGibson action dimension: {og_action_dim}")
        print(f"GR00T robot model joints: {robot_model.num_joints}")
        if og_joint_names:
            print(f"OmniGibson controllable joints ({len(og_joint_names)}): {og_joint_names[:10]}...")
    
    # Store action dimension for adapter
    action_adapter.og_action_dim = og_action_dim
    action_adapter.og_joint_names = og_joint_names

    def _to_robot_action_dim(robot, action: np.ndarray) -> np.ndarray:
        """Ensure action length matches robot.action_dim (trim or zero-pad)."""
        action = np.asarray(action, dtype=np.float32).ravel()
        adim = getattr(robot, "action_dim", None)
        if adim is None:
            return action
        if len(action) > adim:
            return action[:adim].copy()
        if len(action) < adim:
            return np.concatenate([action, np.zeros(adim - len(action), dtype=np.float32)])
        return action

    def _get_controller_order_joint_names(robot):
        """Get joint names in the order expected by env.step(action) (controller order)."""
        if not hasattr(robot, "controller_order") or not hasattr(robot, "_controllers"):
            return None
        # Joint names in articulation/dof order (index i = dof index). UnitreeG1 uses robot.joints, not joint_names/controllable_joints.
        og_names = getattr(robot, "joint_names", None)
        if og_names is None and hasattr(robot, "joints") and robot.joints:
            og_names = list(robot.joints.keys())
        if og_names is None and hasattr(robot, "controllable_joints"):
            og_names = [j.name for j in robot.controllable_joints]
        if og_names is None or len(og_names) == 0:
            return None
        names_in_controller_order = []
        for cname in robot.controller_order:
            controller = robot._controllers[cname]
            dof_idx = controller.dof_idx
            if hasattr(dof_idx, "cpu"):
                dof_idx = dof_idx.cpu().numpy()
            dof_idx = np.asarray(dof_idx).ravel()
            for i in dof_idx:
                idx = int(i)
                if 0 <= idx < len(og_names):
                    names_in_controller_order.append(og_names[idx])
        return names_in_controller_order if names_in_controller_order else None

    def _pinocchio_name_to_index(robot_model, og_name: str):
        """Resolve OmniGibson joint name to Pinocchio q index (exact or fuzzy match)."""
        try:
            return robot_model.dof_index(og_name)
        except (ValueError, KeyError):
            pass
        # Fuzzy: normalize and find best match in robot_model joint names
        og_norm = og_name.lower().replace("_", "").replace("-", "").replace(" ", "")
        for pname in robot_model.joint_names:
            pnorm = pname.lower().replace("_", "").replace("-", "").replace(" ", "")
            if og_norm == pnorm or og_norm in pnorm or pnorm in og_norm:
                try:
                    return robot_model.dof_index(pname)
                except (ValueError, KeyError):
                    continue
        return None

    def _og_jpos_to_pinocchio_q(robot, jpos, robot_model) -> np.ndarray:
        """
        Convert joint positions from OmniGibson articulation order (same as robot.get_joint_positions())
        to Pinocchio order expected by the GR00T/WBC policy. Ensures observation["q"] is interpreted
        with the correct joint-to-index mapping (body_indices, lower_body, etc. are Pinocchio indices).
        """
        jpos = np.asarray(jpos, dtype=np.float64).ravel()
        og_names = getattr(robot, "joint_names", None)
        if og_names is None and hasattr(robot, "joints") and robot.joints:
            og_names = list(robot.joints.keys())
        if og_names is None or len(og_names) == 0:
            return jpos
        n = getattr(robot_model, "num_joints", None) or getattr(robot_model, "num_dofs", len(jpos))
        q_pin = np.zeros(int(n), dtype=np.float64)
        og_name_list = list(og_names)
        for name in getattr(robot_model, "joint_names", []):
            try:
                pidx = robot_model.dof_index(name)
            except (ValueError, KeyError):
                continue
            if pidx < 0 or pidx >= n:
                continue
            # Match name to OG (exact then fuzzy)
            og_idx = None
            if name in og_name_list:
                og_idx = og_name_list.index(name)
            else:
                norm = name.lower().replace("_", "").replace("-", "").replace(" ", "")
                for i, og in enumerate(og_name_list):
                    if og.lower().replace("_", "").replace("-", "").replace(" ", "") == norm:
                        og_idx = i
                        break
            if og_idx is not None and og_idx < len(jpos):
                q_pin[pidx] = float(jpos[og_idx])
        return q_pin.astype(np.float32)

    def _pinocchio_q_to_og_jpos(robot, q_pinocchio: np.ndarray, robot_model) -> np.ndarray:
        """
        Convert joint positions from Pinocchio order to OmniGibson articulation order
        (same as robot.get_joint_positions()). Used to set robot.set_joint_positions() from
        robot_model.default_body_pose so the robot starts in a standing pose.
        """
        q = np.asarray(q_pinocchio, dtype=np.float64).ravel()
        og_names = getattr(robot, "joint_names", None)
        if og_names is None and hasattr(robot, "joints") and robot.joints:
            og_names = list(robot.joints.keys())
        if og_names is None or len(og_names) == 0:
            return q
        jpos_og = np.zeros(len(og_names), dtype=np.float64)
        for og_idx, og_name in enumerate(og_names):
            pidx = _pinocchio_name_to_index(robot_model, og_name)
            if pidx is not None and 0 <= pidx < len(q):
                jpos_og[og_idx] = float(q[pidx])
        return jpos_og.astype(np.float32)

    def _pinocchio_q_to_controller_order_action(robot, q_pinocchio: np.ndarray, robot_model) -> np.ndarray:
        """
        Convert joint positions from Pinocchio (WBC/robot_model) order to the flat action
        vector expected by OmniGibson env.step() (controller order), by matching joint names.
        Use this when q is in Pinocchio order (e.g. WBC output) so pose is correct.
        Output length is trimmed/padded to robot.action_dim (e.g. 29 for g1_29dof_with_hand).
        """
        q = np.asarray(q_pinocchio, dtype=np.float64).ravel()
        names_in_order = _get_controller_order_joint_names(robot)
        if names_in_order is None:
            # Fallback: assume q is already in robot dof order and reorder by controller
            return _to_robot_action_dim(robot, _joint_positions_to_controller_order_action(robot, q))
        out = []
        for og_name in names_in_order:
            pidx = _pinocchio_name_to_index(robot_model, og_name)
            if pidx is not None and 0 <= pidx < len(q):
                out.append(q[pidx])
            else:
                out.append(0.0)
        return _to_robot_action_dim(robot, np.array(out, dtype=np.float32))

    def _joint_positions_to_controller_order_action(robot, q_mapped: np.ndarray) -> np.ndarray:
        """
        Convert joint positions (in robot dof order) to controller-order flat action.
        Only use when q_mapped is already in the same order as robot's internal dof (e.g. from
        adapter output that was built by name mapping). For WBC output (Pinocchio order), use
        _pinocchio_q_to_controller_order_action(robot, q, robot_model) instead.
        """
        q = np.asarray(q_mapped, dtype=np.float64).ravel()
        if not hasattr(robot, "controller_order") or not hasattr(robot, "_controllers"):
            return q
        parts = []
        for name in robot.controller_order:
            controller = robot._controllers[name]
            dof_idx = controller.dof_idx
            if hasattr(dof_idx, "cpu"):
                dof_idx = dof_idx.cpu().numpy()
            dof_idx = np.asarray(dof_idx).ravel()
            if len(dof_idx) == 0:
                continue
            if np.max(dof_idx) >= len(q) or np.min(dof_idx) < 0:
                return q  # fallback: return as-is if indices invalid
            parts.append(q[dof_idx])
        if not parts:
            return q
        return np.concatenate(parts).astype(np.float32)

    def _verify_wbc_omnigibson_mapping(robot, robot_model, verbose: bool):
        """
        Verify that every joint in OmniGibson controller order maps to a valid Pinocchio index.
        Logs mapping summary and warns on any unmapped joints (they receive 0.0 in action).
        """
        names_in_order = _get_controller_order_joint_names(robot)
        if names_in_order is None:
            if verbose:
                print("WBC–OmniGibson mapping: could not get controller-order joint names (skip verification).")
            return
        adim = getattr(robot, "action_dim", None)
        if adim is not None and len(names_in_order) != adim:
            print(
                f"WBC–OmniGibson mapping WARNING: controller-order joints ({len(names_in_order)}) != robot.action_dim ({adim})",
                file=sys.stderr,
            )
        unmapped = []
        mapping_ok = []
        for i, og_name in enumerate(names_in_order):
            pidx = _pinocchio_name_to_index(robot_model, og_name)
            if pidx is None:
                unmapped.append((i, og_name))
            else:
                mapping_ok.append((i, og_name, pidx))
        if unmapped:
            print(
                f"WBC–OmniGibson mapping: {len(unmapped)} joint(s) have no Pinocchio match (will get 0.0): {[n for _, n in unmapped]}",
                file=sys.stderr,
            )
        if verbose and mapping_ok:
            print("WBC–OmniGibson mapping (controller_order -> Pinocchio index):")
            for i, og_name, pidx in mapping_ok[:20]:
                print(f"  [{i}] {og_name} -> q[{pidx}]")
            if len(mapping_ok) > 20:
                print(f"  ... and {len(mapping_ok) - 20} more.")
        if not unmapped and verbose:
            print("WBC–OmniGibson mapping: all controller-order joints map to Pinocchio indices.")

    # Verify WBC <-> OmniGibson joint mapping once at startup
    if env.robots and robot_model is not None:
        _verify_wbc_omnigibson_mapping(env.robots[0], robot_model, args.verbose)

    # Optional: inject standing pose as reset_joint_pos so the robot starts standing (major fix for jump/collapse).
    # G1 has no reset_joint_pos in YAML -> OmniGibson uses zeros -> unstable; hold/zero position/velocity all fail.
    if args.inject_standing_reset and env.robots and robot_model is not None and hasattr(robot_model, "default_body_pose"):
        robot = env.robots[0]
        standing_pinocchio = np.asarray(robot_model.default_body_pose).ravel()
        if len(standing_pinocchio) >= getattr(robot_model, "num_joints", 0) and "robots" in env_config and len(env_config["robots"]) > 0:
            standing_og = _pinocchio_q_to_og_jpos(robot, standing_pinocchio, robot_model)
            try:
                env.close()
            except Exception:
                pass
            # Simulator must be stopped before loading scene when creating a new Environment.
            if hasattr(og, "sim") and og.sim is not None and not og.sim.is_stopped():
                og.sim.stop()
            env_config["robots"][0] = dict(env_config["robots"][0])
            env_config["robots"][0]["reset_joint_pos"] = standing_og.tolist()
            env = og.Environment(configs=env_config)
            print("inject_standing_reset: set reset_joint_pos from robot_model.default_body_pose; recreated env.")

    # Re-enable gravity on all robot links so the robot falls instead of floating (ControllableObject disables it on non-base links).
    if args.enable_full_robot_gravity and env.robots:
        for robot in env.robots:
            if hasattr(robot, "enable_gravity"):
                robot.enable_gravity()
        if args.verbose:
            print("enable_full_robot_gravity: enabled gravity on all robot links.")

    # WBC (Whole-Body Control) for sim2behavior: same as RoboCasa locomanip pipeline
    wbc_policy = None
    concat_action = None
    if args.use_wbc:
        try:
            from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
            from gr00t_wbc.control.policy.wbc_policy_factory import get_wbc_policy
            from gr00t_wbc.control.utils.n1_utils import concat_action as _concat_action
            concat_action = _concat_action
            config = BaseConfig(wbc_version="gear_wbc", enable_waist=True)
            wbc_config = config.load_wbc_yaml()
            wbc_config["upper_body_policy_type"] = "identity"
            robot_type = "g1"
            wbc_policy = get_wbc_policy(robot_type, robot_model, wbc_config)
            wbc_policy.activate_policy()
            print("WBC enabled: lower-body (legs + waist) from G1GearWbcPolicy, upper-body from GR00T.")
        except Exception as e:
            err_msg = str(e)
            print(f"Failed to setup WBC: {e}", file=sys.stderr)
            if "onnxruntime" in err_msg or (isinstance(e, ModuleNotFoundError) and getattr(e, "name", "") == "onnxruntime"):
                print("\nWBC requires onnxruntime. Install it with:", file=sys.stderr)
                print("  pip install onnxruntime", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # Setup video saving (optional)
    frames_dir = None
    camera_frames = {}
    if args.save_video:
        frames_dir = os.path.join(SIM2BEHAVIOR_ROOT, "output_frames")
        os.makedirs(frames_dir, exist_ok=True)
        try:
            import imageio
        except ImportError:
            print("Warning: imageio not installed; cannot save video", file=sys.stderr)
            imageio = None
    else:
        imageio = None

    def _sanitize_camera_name(name):
        return str(name).replace(":", "_").replace("/", "_").replace(" ", "_") or "camera"

    def _collect_all_rgb(obs):
        """Recurse obs and yield (camera_name, rgb_array) for every dict that has 'rgb'."""
        if obs is None or not isinstance(obs, dict):
            return
        for k, v in obs.items():
            if isinstance(v, dict):
                if "rgb" in v:
                    rgb = np.asarray(v["rgb"])
                    if rgb.size > 0 and rgb.ndim >= 2 and rgb.shape[-1] in (3, 4):
                        yield (k, rgb[:, :, :3].astype(np.uint8))
                else:
                    for cam_name, rgb in _collect_all_rgb(v):
                        yield (cam_name, rgb)
            elif isinstance(v, np.ndarray) and v.ndim >= 2 and v.shape[-1] in (3, 4):
                yield (k or "camera", np.asarray(v)[:, :, :3].astype(np.uint8))

    # External camera (topdown_camera) when following robot: fixed height 1m, offset 1m in x and y from robot, looking at robot.
    EXTERNAL_CAMERA_HEIGHT = 1.0
    EXTERNAL_CAMERA_OFFSET_XY = (1.0, 1.0)  # (dx, dy) from robot center

    def _quat_mult_xyzw(q1, q2):
        """Multiply two quaternions in (x, y, z, w) order: q1 * q2 (apply q2 then q1 in world frame)."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        return np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ], dtype=np.float32)

    def _look_at_quaternion(camera_pos, target_pos, up=(0.0, 0.0, 1.0)):
        """Return quaternion (x,y,z,w) so that camera at camera_pos looks at target_pos.
        VisionSensor in this stack uses +Z as view direction; so camera +Z = forward (toward target), +Y = up.
        """
        cam = np.asarray(camera_pos, dtype=np.float64).ravel()[:3]
        tgt = np.asarray(target_pos, dtype=np.float64).ravel()[:3]
        up = np.asarray(up, dtype=np.float64).ravel()[:3]
        forward = tgt - cam
        n = np.linalg.norm(forward)
        if n < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        forward /= n
        # VisionSensor view = +Z. So camera +Z must point at robot -> third column = +forward. Up = world Z-ish.
        right = np.cross(up, forward)
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right /= rn
        up_cam = np.cross(forward, right)
        # R: columns = camera X, Y, Z in world. View = +Z so column 3 = +forward.
        R = np.column_stack([right, up_cam, forward])
        # Rotation matrix to quaternion (x, y, z, w)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=np.float32)
        return q / np.linalg.norm(q)

    def _update_topdown_follow_robot(step_for_log=0):
        """Update external camera: height 1m, offset 1m in x and y from robot, oriented at robot center (when --topdown_follow_robot)."""
        if not args.topdown_follow_robot:
            return
        external = env.external_sensors if env else None
        if not external or "topdown_camera" not in external:
            return
        if not env.robots:
            return
        robot = env.robots[0]
        robot_pos, robot_orn = robot.get_position_orientation(frame="world")
        if hasattr(robot_pos, "cpu"):
            robot_pos = robot_pos.cpu().numpy()
        else:
            robot_pos = np.asarray(robot_pos)
        # Camera at height 1m, 1m offset in x and y from robot center
        camera_pos = np.array([
            float(robot_pos[0]) + EXTERNAL_CAMERA_OFFSET_XY[0],
            float(robot_pos[1]) + EXTERNAL_CAMERA_OFFSET_XY[1],
            EXTERNAL_CAMERA_HEIGHT,
        ], dtype=np.float32)
        # Orient camera so view axis (+Z) points at robot; single look-at.
        camera_orn = _look_at_quaternion(camera_pos, robot_pos)
        # Camera was opposite: rotate 180° around world Z.
        camera_orn = _quat_mult_xyzw(
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),  # 180° around Z
            camera_orn,
        )
        external["topdown_camera"].set_position_orientation(
            position=camera_pos, orientation=camera_orn, frame="world"
        )
        if args.verbose and step_for_log == 0:
            print(f"[step 0] robot_pos(world)={robot_pos.tolist()}, external_camera_pos={camera_pos.tolist()}, look_at_robot")

    # Main loop
    if args.topdown_follow_robot:
        print("Topdown camera will follow robot (height 1m, offset 1m in x/y, oriented at robot center)")
    print(f"Starting simulation loop (max {args.max_steps} steps)...")
    reset_out = env.reset()
    og_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    current_q = None  # Track current joint state for action adapter

    # initial_standing_pose for --hold_initial_pose is captured from the sim at step 0 only (see loop below).
    # We do NOT use robot_model.default_body_pose here: it does not match OmniGibson's G1 (causes floating, legs backwards, flying).
    initial_standing_pose = None

    # Dump camera params (D435, mid360, topdown, etc.) when DUMP_CAMERAS=1
    if os.environ.get("DUMP_CAMERAS", "").lower() in ("1", "true", "t"):
        _dump_camera_params(og, frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames"))

    # Setup per-frame camera extrinsics dumping
    camera_extrinsics_log = []
    if args.dump_camera_extrinsics_per_frame:
        frames_dir = frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames")
        os.makedirs(frames_dir, exist_ok=True)
        print("Per-frame camera extrinsics dumping enabled (D435 and topdown cameras)")

    # Action dump: single text file (JSON Lines) with raw GR00T, WBC goal, WBC output per step
    dump_actions_file = None
    dump_actions_path = None
    if args.dump_actions:
        dump_actions_dir = args.dump_actions_dir or os.path.join(
            frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames"),
            "action_dumps",
        )
        os.makedirs(dump_actions_dir, exist_ok=True)
        dump_actions_path = os.path.join(dump_actions_dir, "actions_dump.jsonl")
        dump_actions_file = open(dump_actions_path, "w", buffering=1)
        print(f"Action dumps enabled: writing to {dump_actions_path}")

    # initial_standing_pose already set above when --hold_initial_pose (from default_body_pose); else None

    for step in range(args.max_steps):
        if step % 100 == 0:
            print(f"Step {step} of {args.max_steps}")

        # Update topdown camera to follow robot (before step so next obs uses updated pose)
        _update_topdown_follow_robot(step_for_log=step)

        # Dump camera extrinsics for this frame (before step, so we capture pose before action)
        if args.dump_camera_extrinsics_per_frame:
            frame_extrinsics = _get_camera_extrinsics_per_frame(og, camera_names=["d435", "topdown"])
            if frame_extrinsics:
                camera_extrinsics_log.append({
                    "step": step,
                    "cameras": frame_extrinsics
                })

        # OmniGibson returns proprio as a concatenated vector, not a dict with joint_qpos/joint_velocities.
        # Enrich obs with joint positions, joint velocities, and (for WBC) base pose/velocity.
        jpos_og_raw = None  # OG-order joint positions (for --hold_initial_pose capture, step 0 only)
        if robot_name_in_config in og_obs and env.robots:
            robot = env.robots[0]
            jpos = robot.get_joint_positions()
            if hasattr(jpos, "cpu"):
                jpos = jpos.detach().cpu().numpy()
            else:
                jpos = np.asarray(jpos)
            # Keep raw OG-order copy for hold_initial_pose / no_op_first_step so we send exactly what the sim expects (no Pinocchio mapping).
            if step == 0 and (args.hold_initial_pose or args.no_op_first_step):
                jpos_og_raw = np.asarray(jpos).ravel().copy()
            # Convert to Pinocchio order so policy/WBC see correct joint-to-index mapping (body_indices, etc.)
            if robot_model is not None:
                jpos = _og_jpos_to_pinocchio_q(robot, jpos, robot_model)
            if isinstance(og_obs[robot_name_in_config], dict):
                og_obs[robot_name_in_config]["joint_qpos"] = jpos
                # Joint velocities: same OG->Pinocchio mapping so adapter/WBC get observation["dq"] and dq_body_scaled != 0
                try:
                    jvel = robot.get_joint_velocities()
                    if hasattr(jvel, "cpu"):
                        jvel = jvel.detach().cpu().numpy()
                    else:
                        jvel = np.asarray(jvel)
                    if robot_model is not None:
                        jvel = _og_jpos_to_pinocchio_q(robot, jvel, robot_model)
                    og_obs[robot_name_in_config]["joint_velocities"] = jvel.astype(np.float32)
                except Exception:
                    pass
                # Base pose/vel for WBC and policy (adapter uses robot_pos, robot_quat, etc.)
                try:
                    pos, ori = robot.get_position_orientation(frame="world")
                    pos = pos.detach().cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos)
                    ori = ori.detach().cpu().numpy() if hasattr(ori, "cpu") else np.asarray(ori)
                    og_obs[robot_name_in_config]["robot_pos"] = pos[:3]
                    og_obs[robot_name_in_config]["robot_quat"] = ori[:4]  # x,y,z,w
                    lin = robot.get_linear_velocity(frame="world")
                    ang = robot.get_angular_velocity(frame="world")
                    lin = lin.detach().cpu().numpy() if hasattr(lin, "cpu") else np.asarray(lin)
                    ang = ang.detach().cpu().numpy() if hasattr(ang, "cpu") else np.asarray(ang)
                    og_obs[robot_name_in_config]["robot_lin_vel"] = lin[:3]
                    og_obs[robot_name_in_config]["robot_ang_vel"] = ang[:3]
                except Exception:
                    pass

        # Adapt OmniGibson observation to GR00T format
        gr00t_obs = obs_adapter.adapt(og_obs, verbose=(args.verbose and step == 0))
        # Use current observation's q for action adapter (always from latest state)
        current_q = gr00t_obs["q"].copy()
        # Build step-0 no-op action (current pose) so first transition doesn't use policy (--no_op_first_step).
        step0_no_op_action = None
        if args.no_op_first_step and step == 0 and env.robots and robot_model is not None:
            robot = env.robots[0]
            if jpos_og_raw is not None and len(jpos_og_raw) > 0:
                step0_no_op_action = {
                    robot_name_in_config: _to_robot_action_dim(
                        robot, _joint_positions_to_controller_order_action(robot, jpos_og_raw)
                    ).copy()
                }
            else:
                q0 = np.asarray(current_q, dtype=np.float64).ravel()
                if len(q0) >= robot_model.num_joints:
                    step0_no_op_action = {
                        robot_name_in_config: _pinocchio_q_to_controller_order_action(robot, q0, robot_model).copy()
                    }
        # Capture initial standing pose once at step 0 for --hold_initial_pose.
        # Use simulator joint positions in OG order -> controller order (no Pinocchio) so the action
        # is exactly what the sim expects and the robot holds still. Pinocchio-based capture can
        # mis-map (base vs joints, name mismatches) and cause the robot to jump.
        if args.hold_initial_pose and step == 0 and env.robots and initial_standing_pose is None:
            robot = env.robots[0]
            if jpos_og_raw is not None and len(jpos_og_raw) > 0:
                initial_standing_pose = {
                    robot_name_in_config: _to_robot_action_dim(
                        robot, _joint_positions_to_controller_order_action(robot, jpos_og_raw)
                    ).copy()
                }
                if args.verbose:
                    print("hold_initial_pose: captured initial standing pose from step-0 sim joint positions (OG->controller order).")
            elif robot_model is not None:
                q0 = np.asarray(current_q, dtype=np.float64).ravel()
                if len(q0) >= robot_model.num_joints:
                    initial_standing_pose = {
                        robot_name_in_config: _pinocchio_q_to_controller_order_action(robot, q0, robot_model).copy()
                    }
                    if args.verbose:
                        print("hold_initial_pose: captured initial standing pose from step-0 observation (Pinocchio fallback).")
        if args.verbose and step == 0:
            print(f"GR00T obs keys: {list(gr00t_obs.keys())}")
            print(f"Camera images: {[k for k in gr00t_obs.keys() if 'image' in k or 'video' in k]}")

        # Get action from policy
        try:
            gr00t_action, info = policy.get_action(gr00t_obs)
        except Exception as e:
            print(f"Error getting action from policy at step {step}: {e}", file=sys.stderr)
            break

        # Snapshot raw GR00T output for action dump (before WBC may overwrite gr00t_action)
        gr00t_action_for_dump = None
        wbc_goal_for_dump = None
        wbc_action_for_dump = None
        if dump_actions_file is not None:
            gr00t_action_for_dump = copy.deepcopy(gr00t_action)
            for k in list(gr00t_action_for_dump.keys()):
                v = gr00t_action_for_dump[k]
                if hasattr(v, "cpu"):
                    gr00t_action_for_dump[k] = np.asarray(v.cpu().numpy())
                else:
                    gr00t_action_for_dump[k] = np.asarray(v)

        # Apply WBC (same as RoboCasa locomanip): GR00T -> concat_action -> WBC policy -> full q
        if wbc_policy is not None and concat_action is not None:
            try:
                wbc_goal = concat_action(robot_model, gr00t_action)
                # Take current timestep only: policy returns (B, T, D), WBC expects 1D per step
                for k in list(wbc_goal.keys()):
                    v = wbc_goal[k]
                    if hasattr(v, "shape") and hasattr(v, "reshape"):
                        v = np.asarray(v)
                        if v.ndim == 3:
                            v = v[0, 0, :]   # (B, T, D) -> (D,)
                        elif v.ndim == 2:
                            v = v[0, :]      # (T, D) or (B, D) -> (D,)
                        if v.ndim > 1:
                            v = v.reshape(-1)
                        wbc_goal[k] = np.asarray(v, dtype=np.float64)
                if dump_actions_file is not None:
                    wbc_goal_for_dump = {k: np.asarray(v).reshape(-1) for k, v in wbc_goal.items()}
                wbc_policy.set_observation(gr00t_obs)
                wbc_policy.set_goal(wbc_goal)
                wbc_action = wbc_policy.get_action()
                if dump_actions_file is not None:
                    wbc_action_for_dump = {"q": np.asarray(wbc_action["q"]).reshape(-1)}
                # Full q from WBC (43-DOF); map to OmniGibson and step
                gr00t_action = {"q": np.asarray(wbc_action["q"]).astype(np.float32)}
            except Exception as e:
                print(f"WBC step error at step {step}: {e}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()

        # Adapt GR00T action to OmniGibson format
        if args.verbose and step == 0:
            print(f"GR00T action keys: {list(gr00t_action.keys())}")
            print(f"OmniGibson action_space type: {type(env.action_space)}")
            # Check action_space structure
            if hasattr(env.action_space, 'spaces'):
                print(f"Action space keys: {list(env.action_space.spaces.keys()) if hasattr(env.action_space.spaces, 'keys') else 'N/A'}")
        
        try:
            if "q" in gr00t_action and gr00t_action.get("q") is not None and len(gr00t_action["q"]) == robot_model.num_joints:
                # Full q already (e.g. from WBC); no need for current_q
                og_action = action_adapter.adapt(gr00t_action, action_space=env.action_space)
            else:
                og_action = action_adapter.adapt_with_current_state(
                    gr00t_action, current_q, action_space=env.action_space
                )
        except Exception as e:
            if args.verbose:
                print(f"Error adapting action with current state: {e}", file=sys.stderr)
            # Fallback: try without current state
            try:
                og_action = action_adapter.adapt(gr00t_action, action_space=env.action_space)
            except Exception as e2:
                print(f"Error adapting action (fallback): {e2}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                break
        
        # Convert to OmniGibson controller order so env.step() applies correct joint targets.
        # When we have full q from WBC (Pinocchio order), map by joint name to controller order.
        if (
            isinstance(og_action, dict)
            and env.robots
            and robot_name_in_config in og_action
            and "q" in gr00t_action
            and gr00t_action.get("q") is not None
            and len(gr00t_action["q"]) == robot_model.num_joints
        ):
            robot = env.robots[0]
            q_pinocchio = np.asarray(gr00t_action["q"], dtype=np.float64).ravel()
            if q_pinocchio.size == getattr(robot, "n_dof", q_pinocchio.size):
                # WBC output is in Pinocchio order: map by joint name to controller order
                og_action[robot_name_in_config] = _pinocchio_q_to_controller_order_action(
                    robot, q_pinocchio, robot_model
                )
        elif isinstance(og_action, dict) and env.robots and robot_name_in_config in og_action:
            robot = env.robots[0]
            q_vec = np.asarray(og_action[robot_name_in_config], dtype=np.float64).ravel()
            if q_vec.size == getattr(robot, "n_dof", q_vec.size):
                # Adapter output in dof order: reorder by controller
                og_action[robot_name_in_config] = _to_robot_action_dim(
                    robot, _joint_positions_to_controller_order_action(robot, q_vec)
                )

        # Optionally send initial standing pose every step (robot holds still)
        if args.hold_initial_pose and initial_standing_pose is not None:
            og_action = copy.deepcopy(initial_standing_pose)

        # On step 0 only: send current pose (no-op) so first transition never uses policy (avoids snap).
        if step == 0 and args.no_op_first_step and step0_no_op_action is not None:
            og_action = copy.deepcopy(step0_no_op_action)

        # Optional: clamp sent_to_simulator indices to fixed values (e.g. 4 and 10 to 0.1, 14 to 0.01) to test snap.
        if (
            (args.clamp_sent_indices_4_10 is not None or args.clamp_sent_index_14 is not None)
            and isinstance(og_action, dict)
            and robot_name_in_config in og_action
        ):
            arr = np.asarray(og_action[robot_name_in_config], dtype=np.float64).copy()
            if args.clamp_sent_indices_4_10 is not None and len(arr) > 10:
                arr[4] = arr[10] = float(args.clamp_sent_indices_4_10)
            if args.clamp_sent_index_14 is not None and len(arr) > 14:
                arr[14] = float(args.clamp_sent_index_14)
            og_action[robot_name_in_config] = arr

        # Optional: zero the 15 base (lower-body) joints (for use with base use_delta_commands: true to hold pose).
        # When zero_base_action is not set, the first 15 values come from WBC lower-body (real ONNX/WBC actions).
        if args.zero_base_action and isinstance(og_action, dict) and robot_name_in_config in og_action:
            arr = np.asarray(og_action[robot_name_in_config], dtype=np.float64).copy()
            if len(arr) >= 15:
                arr[0:15] = 0.0
                og_action[robot_name_in_config] = arr

        # Write per-step action dump to single text file (JSON Lines: one JSON object per step)
        if dump_actions_file is not None and gr00t_action_for_dump is not None:
            # On first step, write controller-order joint names so sent_to_simulator indices can be labeled
            if step == 0 and env.robots:
                try:
                    robot = env.robots[0]
                    names_in_order = _get_controller_order_joint_names(robot)
                    if names_in_order and dump_actions_path is not None:
                        names_path = os.path.join(os.path.dirname(dump_actions_path), "controller_order_joint_names.txt")
                        with open(names_path, "w") as f:
                            f.write("# sent_to_simulator index -> joint name (controller order)\n")
                            for i, name in enumerate(names_in_order):
                                f.write(f"{i}\t{name}\n")
                        if args.verbose:
                            print(f"Wrote controller-order joint names to {names_path}")
                except Exception as e:
                    if args.verbose:
                        print(f"Could not write controller-order joint names: {e}", file=sys.stderr)
            step_dict = {"step": step}
            if args.hold_initial_pose:
                step_dict["hold_initial_pose"] = True
            if step == 0 and args.no_op_first_step:
                step_dict["no_op_first_step"] = True
            gr00t_serial = {}
            _T = 30  # policy action horizon
            for k, v in gr00t_action_for_dump.items():
                arr = np.asarray(v).reshape(-1)
                if k == "action.base_height_command":
                    # Dump only the first timestep value (t=0); ignore t=1,2,... predictions
                    arr = arr[:1] if arr.size > 0 else arr
                elif arr.size > _T and arr.size % _T == 0:
                    d = arr.size // _T
                    arr = arr[:d]
                gr00t_serial[k] = arr.tolist()
            step_dict["gr00t"] = gr00t_serial
            if wbc_goal_for_dump is not None:
                step_dict["wbc_goal"] = {
                    k: np.asarray(v).reshape(-1).tolist() for k, v in wbc_goal_for_dump.items()
                }
            if wbc_action_for_dump is not None:
                step_dict["wbc"] = {
                    k: np.asarray(v).reshape(-1).tolist() for k, v in wbc_action_for_dump.items()
                }
                # Explicit lower/upper body slices so lower-body WBC output is obvious (G1: 15 + 28)
                q_full = np.asarray(wbc_action_for_dump["q"]).reshape(-1)
                if len(q_full) >= 43:
                    step_dict["wbc"]["q_lower_body"] = q_full[:15].tolist()
                    step_dict["wbc"]["q_upper_body"] = q_full[15:43].tolist()
                # Dump 86-D ONNX single-step input with labeled keys (cmd_scaled, height_cmd, etc.)
                if wbc_policy is not None and hasattr(wbc_policy, "lower_body_policy"):
                    lb = getattr(wbc_policy, "lower_body_policy", None)
                    if lb is not None and hasattr(lb, "get_last_single_obs_dict"):
                        onnx_86 = lb.get_last_single_obs_dict()
                        if onnx_86 is not None:
                            step_dict["onnx_input_86"] = onnx_86
            # Include the exact action sent to the simulator
            step_dict["sent_to_simulator"] = _tensor_to_python(copy.deepcopy(og_action))
            # Format all numbers as 6-character strings
            step_dict = _format_dump_6(step_dict)
            try:
                dump_actions_file.write(json.dumps(step_dict) + "\n")
                dump_actions_file.flush()
            except Exception as e:
                if args.verbose:
                    print(f"Action dump write failed for step {step}: {e}", file=sys.stderr)

        # The adapter should return the correct format; ensure it's valid
        if args.verbose and step == 0:
            if isinstance(og_action, dict):
                print(f"OmniGibson action type: dict, keys: {list(og_action.keys())}")
                for k, v in og_action.items():
                    print(f"  {k}: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}")
            else:
                print(f"OmniGibson action type: {type(og_action)}, shape: {og_action.shape if hasattr(og_action, 'shape') else 'N/A'}")

        # Step environment
        result = env.step(og_action)
        if len(result) == 5:
            og_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            og_obs, reward, done, info = result

        # current_q is updated at the start of the next iteration from the new og_obs

        # Save video frames (optional)
        if args.save_video:
            for cam_name, rgb in _collect_all_rgb(og_obs):
                if cam_name not in camera_frames:
                    camera_frames[cam_name] = []
                camera_frames[cam_name].append(rgb)

        if done:
            print(f"Episode done at step {step}")
            reset_out = env.reset()
            og_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            policy.reset()

    if dump_actions_file is not None:
        dump_actions_file.close()
        print(f"Action dumps written to {dump_actions_path}")

    # Save videos
    if args.save_video and camera_frames and imageio:
        for cam_name, frames in camera_frames.items():
            if not frames:
                continue
            safe_name = _sanitize_camera_name(cam_name)
            out_path = os.path.join(frames_dir, f"{safe_name}_gr00t.mp4")
            try:
                imageio.mimsave(out_path, frames, fps=args.video_fps)
                print(f"Saved video: {out_path}")
            except Exception as e:
                print(f"Could not write video {out_path}: {e}", file=sys.stderr)
    
    # Save per-frame camera extrinsics log
    if args.dump_camera_extrinsics_per_frame and camera_extrinsics_log:
        extrinsics_path = os.path.join(frames_dir, "camera_extrinsics_per_frame.json")
        try:
            with open(extrinsics_path, "w") as f:
                json.dump(camera_extrinsics_log, f, indent=2)
            print(f"\nSaved per-frame camera extrinsics: {extrinsics_path} ({len(camera_extrinsics_log)} frames)")
            print(f"  Cameras logged: {set(name for entry in camera_extrinsics_log for name in entry['cameras'].keys())}")
        except Exception as e:
            print(f"Failed to save camera extrinsics: {e}", file=sys.stderr)

    print("Simulation complete.")
    
    # Segfault (exit 139) often happens here: env.close() tears down Isaac Sim (render, PhysX).
    # Headless teardown can trigger driver/sim bugs. Set OMNIGIBSON_SKIP_CLOSE=1 to skip close
    # and confirm (process will exit 0 but resources may leak).
    if not os.environ.get("OMNIGIBSON_SKIP_CLOSE"):
        try:
            env.close()
        except Exception as e:
            print(f"Warning: Error during env.close(): {e}", file=sys.stderr)
            print("This is often a segfault in Isaac Sim teardown. Set OMNIGIBSON_SKIP_CLOSE=1 to skip.", file=sys.stderr)
    else:
        print("Skipping env.close() (OMNIGIBSON_SKIP_CLOSE=1)", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
