#!/usr/bin/env python3
"""
Minimal OmniGibson + Unitree G1 standalone script (sim2behavior).
Loads G1 from sim2behavior resources, creates a minimal scene, runs step loop with zero or keyboard action.
Use a dedicated BEHAVIOR/OmniGibson venv; see sim2behavior_DESIGN.md §11.

For GPUs without raytracing (e.g. A5000), run with headless + RTX disabled (set below before og import).

If you get segmentation fault (exit code 139): headless + XR/RTX issues are common. This script
disables XR extensions by default. The crash often happens in env.close() (teardown). Try
OMNIGIBSON_SKIP_CLOSE=1 to confirm (then exit 0 without closing; resources may leak). Or run with
a display (DISPLAY=:0, OMNIGIBSON_HEADLESS=0) to see if the crash is headless-only.

Visualization:
  - With display: DISPLAY=:0 OMNIGIBSON_HEADLESS=0 python ... (OmniGibson may open a viewport).
  - Headless: set SAVE_VIDEO=1 to write frames to sim2behavior/output_frames/ and print the path.

Why no rgb in obs? The URDF has camera links (d435_link, mid360_link), but UnitreeG1 loads from
USD (omnigibson-robot-assets). OmniGibson discovers vision from Camera prims in that USD; the
import script must add them (camera_links: ["d435_link", "mid360_link"]). If the G1 USD was built
without that step, obs has only proprio. Re-import the G1 asset with cameras, or use env.render().
"""
from __future__ import annotations

import os
import sys
import warnings

# Suppress noisy Gymnasium warning when env casts action/obs to numpy (harmless).
warnings.filterwarnings("ignore", message="Casting input x to numpy array", module="gymnasium.spaces.box")

# Optional: point to dataset root so UnitreeG1 loads a USD that has Camera prims (for rgb in obs).
# UnitreeG1 loads from {OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/models/unitree_g1/usd/unitree_g1.usda
# Set before importing omnigibson, e.g.: export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
# See docs/USD_CAMERAS_AND_CONFIG.md for using a custom G1 USD with cameras.

# Headless + disable RTX and XR (required before any omnigibson import). Use for A5000 / no raytracing / servers.
# XR extensions can cause segmentation faults (exit 139) in headless; add flags before og is imported.
os.environ["OMNIGIBSON_HEADLESS"] = "1"
os.environ["ISAAC_SIM_HEADLESS"] = "1"
os.environ["DISPLAY"] = ""
os.environ["RTX_DRIVER_VERIFICATION"] = "0"
_isaac_args = [
    "--/rtx/verifyDriverVersion/enabled=false",
    "--/rtx/enabled=false",
    "--/app/extensions/omni.kit.xr.core/disabled=true",
    "--/app/extensions/omni.kit.xr.profile.vr/disabled=true",
    "--/app/extensions/omni.kit.xr/disabled=true",
]
for arg in _isaac_args:
    if arg not in sys.argv:
        sys.argv.append(arg)

# Resolve sim2behavior resources path (script is in sim2behavior/scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM2BEHAVIOR_ROOT = os.path.dirname(SCRIPT_DIR)
G1_RESOURCES = os.path.join(SIM2BEHAVIOR_ROOT, "resources", "robots", "g1")
CONFIG_PATH = os.path.join(G1_RESOURCES, "g1_omnigibson_config.yaml")


def _tensor_to_python(obj):
    """Convert torch/numpy to native Python for YAML/JSON serialization."""
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
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return str(obj)


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
        with open(out_path, "w") as f:
            yaml.safe_dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"Dumped camera params: {out_path}")
    except Exception as e:
        print(f"DUMP_CAMERAS=1: could not write {out_path}: {e}", file=sys.stderr)


def _check_config_and_paths() -> bool:
    """Validate config and resource paths without importing OmniGibson."""
    import yaml

    if not os.path.isfile(CONFIG_PATH):
        print(f"Config not found: {CONFIG_PATH}", file=sys.stderr)
        return False
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    urdf_rel = config.get("robot", {}).get("urdf_path", "urdf/g1_29dof_with_hand.urdf")
    urdf_path = os.path.join(G1_RESOURCES, os.path.normpath(urdf_rel))
    if not os.path.isfile(urdf_path):
        print(f"URDF not found: {urdf_path}", file=sys.stderr)
        print("Create meshes symlink: cd resources/robots/g1 && ln -snf ../../../../sim2mujoco/resources/robots/g1/meshes meshes", file=sys.stderr)
        return False
    meshes_dir = os.path.join(G1_RESOURCES, config.get("robot", {}).get("meshes_dir", "meshes"))
    if not os.path.isdir(meshes_dir):
        print(f"Meshes dir not found: {meshes_dir}", file=sys.stderr)
        print("From sim2behavior: cd resources/robots/g1 && ln -snf ../../../../sim2mujoco/resources/robots/g1/meshes meshes", file=sys.stderr)
        return False
    return True


def main() -> int:
    if not _check_config_and_paths():
        return 1

    try:
        import omnigibson as og
    except ImportError:
        print("OmniGibson not installed. Install per https://behavior.stanford.edu/omnigibson/", file=sys.stderr)
        print("Config and paths validated successfully; run with OmniGibson venv to start sim.", file=sys.stderr)
        print(f"Current Python: {sys.executable}", file=sys.stderr)
        print("If using conda env 'behavior' on shared storage, run: conda activate behavior", file=sys.stderr)
        print("  or: /path/to/miniconda3/envs/behavior/bin/python ...", file=sys.stderr)
        return 0

    # Load config and resolve URDF path for OmniGibson
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    urdf_rel = config["robot"]["urdf_path"]
    urdf_abs = os.path.join(G1_RESOURCES, os.path.normpath(urdf_rel))

    # Create minimal OmniGibson env with G1 (API may vary by OmniGibson version).
    # If no scene config exists, validate and exit 0 so users can integrate manually.
    configs_path = os.path.join(SIM2BEHAVIOR_ROOT, "configs", "g1_standalone.yaml")
    if not os.path.isfile(configs_path):
        print("No configs/g1_standalone.yaml found; skipping env creation.", file=sys.stderr)
        print("G1 URDF for manual use:", urdf_abs, file=sys.stderr)
        print("Create configs/g1_standalone.yaml pointing to this URDF for a minimal OmniGibson scene.", file=sys.stderr)
        return 0

    # Load and resolve URDF path in config (replace ${SIM2BEHAVIOR_ROOT} if present)
    with open(configs_path) as f:
        env_config = yaml.safe_load(f)
    if env_config.get("robots") and len(env_config["robots"]) > 0:
        robot_cfg = env_config["robots"][0]
        if "urdf_path" in robot_cfg:
            robot_cfg["urdf_path"] = robot_cfg["urdf_path"].replace("${SIM2BEHAVIOR_ROOT}", SIM2BEHAVIOR_ROOT)
            # If still relative, make absolute
            if not os.path.isabs(robot_cfg["urdf_path"]):
                robot_cfg["urdf_path"] = os.path.join(SIM2BEHAVIOR_ROOT, robot_cfg["urdf_path"])

    try:
        # OmniGibson Environment() takes only configs= ; action/physics freq are in env_config["env"]
        env = og.Environment(configs=env_config)
    except Exception as e:
        print("OmniGibson Environment() failed:", e, file=sys.stderr)
        print("G1 URDF path for manual use:", urdf_abs, file=sys.stderr)
        return 0

    # Step loop: zero action (OmniGibson action_space can be Dict, so zero each entry).
    # Expected warnings you may see: "Inferring finger link... parallel jaw gripper" (G1 has
    # dexterous hands); "Replicator:Annotators" (Isaac Sim). Both are harmless.
    import numpy as np
    max_steps = int(os.environ.get("OMNIGIBSON_MAX_STEPS", "1000"))
    save_video = os.environ.get("SAVE_VIDEO", "").lower() in ("1", "true", "t")
    frames_dir = os.path.join(SIM2BEHAVIOR_ROOT, "output_frames") if save_video else None
    imageio = None
    if save_video:
        os.makedirs(frames_dir, exist_ok=True)
        try:
            import imageio
        except ImportError:
            pass
        # Per-camera frame lists: camera_name -> list of rgb arrays
        camera_frames = {}

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

    dump_cameras = os.environ.get("DUMP_CAMERAS", "").lower() in ("1", "true", "t")
    if dump_cameras:
        os.makedirs(frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames"), exist_ok=True)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    for step in range(max_steps):
        if step % 100 == 0:
            print(f"Step {step} of {max_steps}")
        if save_video and step % 5 == 0:
            for cam_name, rgb in _collect_all_rgb(obs):
                if cam_name not in camera_frames:
                    camera_frames[cam_name] = []
                camera_frames[cam_name].append(rgb)
        sample = env.action_space.sample()
        if isinstance(sample, dict):
            action = {k: (np.asarray(v) * 0.0) for k, v in sample.items()}
        else:
            action = np.asarray(sample) * 0.0
        result = env.step(action)
        # Gymnasium returns (obs, reward, terminated, truncated, info); older gym returns (obs, reward, done, info)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        # Dump camera intrinsics and extrinsics once (after first step so sensors are initialized).
        if dump_cameras and step == 1:
            _dump_camera_params(og, frames_dir or os.path.join(SIM2BEHAVIOR_ROOT, "output_frames"))

        if done:
            print(f"Done at step {step} of {max_steps}: Episode done")
            print(f"Info: {info}")
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    if save_video and camera_frames:
        if imageio is not None:
            for cam_name, frames in camera_frames.items():
                if not frames:
                    continue
                safe_name = _sanitize_camera_name(cam_name)
                out_path = os.path.join(frames_dir, f"{safe_name}.mp4")
                try:
                    fps = min(10, max(1, len(frames) // 10))
                    imageio.mimsave(out_path, frames, fps=fps)
                    print(f"Saved video: {out_path}")
                except Exception as e:
                    print(f"Could not write video {out_path}: {e}", file=sys.stderr)
                    for i, f in enumerate(frames[:50]):
                        imageio.imwrite(os.path.join(frames_dir, f"{safe_name}_frame_{i:04d}.png"), f)
                    print(f"Saved first 50 frames as PNGs in {frames_dir}")
        else:
            print("SAVE_VIDEO=1 but imageio not installed; install imageio to save frames.", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()

    # Segfault (exit 139) often happens here: env.close() tears down Isaac Sim (render, PhysX).
    # Headless teardown can trigger driver/sim bugs. Set OMNIGIBSON_SKIP_CLOSE=1 to skip close
    # and confirm (process will exit 0 but resources may leak).
    if not os.environ.get("OMNIGIBSON_SKIP_CLOSE"):
        env.close()
    else:
        print("Skipping env.close() (OMNIGIBSON_SKIP_CLOSE=1)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
