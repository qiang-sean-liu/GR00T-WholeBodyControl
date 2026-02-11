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
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

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
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _tensor_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_tensor_to_python(x) for x in obj]
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
                        import numpy as np
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
        default=1,
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

    # Setup video saving (optional)
    import numpy as np
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

    def _update_topdown_follow_robot(step_for_log=0):
        """Update topdown camera to be above the robot (when --topdown_follow_robot is set)."""
        if not args.topdown_follow_robot:
            return
        external = env.external_sensors if env else None
        if not external or "topdown_camera" not in external:
            return
        if not env.robots:
            return
        robot = env.robots[0]
        robot_pos, robot_orn = robot.get_position_orientation(frame="world")
        # Convert to numpy if torch
        if hasattr(robot_pos, "cpu"):
            robot_pos = robot_pos.cpu().numpy()
        else:
            robot_pos = np.asarray(robot_pos)
        camera_pos = robot_pos + np.array([0.0, 0.0, args.topdown_follow_height], dtype=np.float32)
        # Quaternion (x,y,z,w). identity = look along -Z (down per Isaac). Try --topdown_orientation if view wrong.
        _orn_map = {
            "identity": (0.0, 0.0, 0.0, 1.0),
            "flip_x": (1.0, 0.0, 0.0, 0.0),   # 180° around X
            "flip_y": (0.0, 1.0, 0.0, 0.0),   # 180° around Y
            "flip_z": (0.0, 0.0, 1.0, 0.0),   # 180° around Z
        }
        camera_orn = np.array(_orn_map[args.topdown_orientation], dtype=np.float32)
        external["topdown_camera"].set_position_orientation(
            position=camera_pos, orientation=camera_orn, frame="world"
        )
        if args.verbose and step_for_log == 0:
            print(f"[step 0] robot_pos(world)={robot_pos.tolist()}, topdown_camera_pos={camera_pos.tolist()}, orientation={args.topdown_orientation}")

    # Main loop
    if args.topdown_follow_robot:
        print(f"Topdown camera will follow robot (height offset: {args.topdown_follow_height}m, orientation: {args.topdown_orientation})")
    print(f"Starting simulation loop (max {args.max_steps} steps)...")
    reset_out = env.reset()
    og_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    current_q = None  # Track current joint state for action adapter

    # Dump camera params (D435, mid360, topdown, etc.) when DUMP_CAMERAS=1
    if os.environ.get("DUMP_CAMERAS", "").lower() in ("1", "true", "t"):
        _dump_camera_params(og, frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames"))

    # Setup per-frame camera extrinsics dumping
    camera_extrinsics_log = []
    if args.dump_camera_extrinsics_per_frame:
        frames_dir = frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames")
        os.makedirs(frames_dir, exist_ok=True)
        print("Per-frame camera extrinsics dumping enabled (D435 and topdown cameras)")

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

        # Adapt OmniGibson observation to GR00T format
        gr00t_obs = obs_adapter.adapt(og_obs, verbose=(args.verbose and step == 0))
        if current_q is None:
            current_q = gr00t_obs["q"].copy()
        if args.verbose and step == 0:
            print(f"GR00T obs keys: {list(gr00t_obs.keys())}")
            print(f"Camera images: {[k for k in gr00t_obs.keys() if 'image' in k or 'video' in k]}")

        # Get action from policy
        try:
            gr00t_action, info = policy.get_action(gr00t_obs)
        except Exception as e:
            print(f"Error getting action from policy at step {step}: {e}", file=sys.stderr)
            break

        # Adapt GR00T action to OmniGibson format
        if args.verbose and step == 0:
            print(f"GR00T action keys: {list(gr00t_action.keys())}")
            print(f"OmniGibson action_space type: {type(env.action_space)}")
            # Check action_space structure
            if hasattr(env.action_space, 'spaces'):
                print(f"Action space keys: {list(env.action_space.spaces.keys()) if hasattr(env.action_space.spaces, 'keys') else 'N/A'}")
        
        try:
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
        
        # The adapter should return the correct format, but ensure it's valid
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

        # Update current_q from new observation
        if "q" in gr00t_obs:
            current_q = gr00t_obs["q"].copy()

        # Save video frames (optional)
        if args.save_video and step % 5 == 0:
            for cam_name, rgb in _collect_all_rgb(og_obs):
                if cam_name not in camera_frames:
                    camera_frames[cam_name] = []
                camera_frames[cam_name].append(rgb)

        if done:
            print(f"Episode done at step {step}")
            reset_out = env.reset()
            og_obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            policy.reset()

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
        import json
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
