#!/usr/bin/env python3
"""
Visualize a scene USD file in headless mode with RTX disabled, and save a snapshot as PNG or a video.

Usage:
    # Single snapshot:
    python scripts/visualize_scene_usd.py output_frames/Rs_int_scene.usd -o output_frames/Rs_int_snapshot.png

    # Video by sweeping camera x,y over layout bounds (uses scene layout JSON for range):
    python scripts/visualize_scene_usd.py output_frames/Rs_int_scene.usd --video -o output_frames/Rs_int_sweep.mp4

    # Compose all swept frames into one image (overlap computed via phase correlation):
    python scripts/visualize_scene_usd.py output_frames/Rs_int_scene.usd --compose -o output_frames/Rs_int_compose.png

    # Optional: pass layout JSON explicitly (default: inferred from USD name and OMNIGIBSON_DATA_PATH):
    python scripts/visualize_scene_usd.py output_frames/Rs_int_scene.usd --video --layout-json /path/to/Rs_int_best.json

Requires OmniGibson/Isaac Sim Python (same as run_omnigibson_g1_with_gr00t.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Headless + disable RTX/XR (before any omnigibson import)
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("ISAAC_SIM_HEADLESS", "1")
os.environ.setdefault("DISPLAY", "")
os.environ.setdefault("RTX_DRIVER_VERIFICATION", "0")
# os.environ.setdefault("OMNI_DISABLE_RTX", "1")
# os.environ.setdefault("CARB_DISABLE_RTX", "1")

# Isaac Sim args (add to sys.argv for OmniGibson, but filter from argparse)
_isaac_args = []
# _isaac_args = [
#     "--/rtx/verifyDriverVersion/enabled=false",
#     "--/rtx/enabled=false",
#     "--/app/extensions/omni.kit.xr.core/disabled=true",
#     "--/app/extensions/omni.kit.xr.profile.vr/disabled=true",
#     "--/app/extensions/omni.kit.xr/disabled=true",
# ]
_filtered_argv = [a for a in sys.argv if a not in _isaac_args]
for arg in _isaac_args:
    if arg not in sys.argv:
        sys.argv.append(arg)

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM2BEHAVIOR_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_FRAMES = os.path.join(SIM2BEHAVIOR_ROOT, "output_frames")

# Set OMNIGIBSON_DATA_PATH for asset resolution (scene USD may reference behavior-1k-assets)
_DEFAULT_BEHAVIOR_DATA_PATH = "/mnt/nas26/qiang.liu/BEHAVIOR-1K/datasets"
if "OMNIGIBSON_DATA_PATH" not in os.environ and os.path.isdir(_DEFAULT_BEHAVIOR_DATA_PATH):
    os.environ["OMNIGIBSON_DATA_PATH"] = _DEFAULT_BEHAVIOR_DATA_PATH


def _get_layout_bounds_from_json(layout_json_path, margin=1.0):
    """
    Read scene layout JSON and return (x_min, x_max, y_min, y_max) from all object root_link positions.
    margin: added to each side of the bounding box (meters).
    """
    with open(layout_json_path, "r") as f:
        data = json.load(f)
    state = data.get("state", {})
    registry = state.get("registry", state)
    object_registry = registry.get("object_registry", {})
    xs, ys = [], []
    for obj_name, obj_state in object_registry.items():
        root = obj_state.get("root_link", {})
        pos = root.get("pos", [0, 0, 0])
        if len(pos) >= 2:
            xs.append(float(pos[0]))
            ys.append(float(pos[1]))
    if not xs or not ys:
        return -5.0, 5.0, -5.0, 5.0  # fallback
    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin
    return x_min, x_max, y_min, y_max


def _infer_layout_json_path(usd_path, behavior_data_path):
    """
    Infer layout JSON path from USD path: e.g. Rs_int_scene.usd -> scenes/Rs_int/json/Rs_int_best.json.
    behavior_data_path: root that contains datasets/behavior-1k-assets (or behavior-1k-assets).
    """
    base = os.path.splitext(os.path.basename(usd_path))[0]
    # Rs_int_scene -> Rs_int, Beechwood_0_int_scene -> Beechwood_0_int
    if base.endswith("_scene"):
        scene_model = base[: -len("_scene")]
    else:
        scene_model = base
    # OMNIGIBSON_DATA_PATH may be .../datasets or ... (parent of behavior-1k-assets)
    root = behavior_data_path.rstrip("/")
    if os.path.basename(root) == "datasets":
        assets = os.path.join(root, "behavior-1k-assets")
    else:
        assets = os.path.join(root, "datasets", "behavior-1k-assets")
    json_path = os.path.join(assets, "scenes", scene_model, "json", f"{scene_model}_best.json")
    return json_path if os.path.isfile(json_path) else None


def _compose_frames_into_image(frames, nx, ny):
    """
    Compose a grid of frames (nx * ny) into one image by estimating pixel overlap
    between adjacent frames using phase correlation, then placing and blending.
    frames: list of RGB uint8 (H, W, 3), row-major order (row 0: j=0..ny-1, row 1: ...).
    Returns: single RGB uint8 image.
    """
    import cv2
    import numpy as np

    H, W = frames[0].shape[:2]
    # Grayscale float for phase correlation
    def to_gs(f):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        return np.float32(g)

    # Shift from frame (i,j) to frame (i,j+1): shift_x[i,j] = (dx, dy) so that
    # frames[flat(i,j+1)](x+dx, y+dy) aligns with frames[flat(i,j)](x,y)
    def flat(i, j):
        return i * ny + j

    # Phase correlation between two grayscale float images; returns (dx, dy) shift of img2 relative to img1
    def get_shift(img1_gs, img2_gs):
        (dx, dy), _ = cv2.phaseCorrelate(img1_gs, img2_gs)
        return dx, dy

    # Compute shifts between adjacent frames (in x: same row, next col; in y: next row, same col)
    shift_right = {}  # (i, j) -> (dx, dy) from (i,j) to (i, j+1)
    shift_down = {}   # (i, j) -> (dx, dy) from (i,j) to (i+1, j)
    for i in range(nx):
        for j in range(ny - 1):
            a, b = flat(i, j), flat(i, j + 1)
            dx, dy = get_shift(to_gs(frames[a]), to_gs(frames[b]))
            shift_right[(i, j)] = (dx, dy)
    for i in range(nx - 1):
        for j in range(ny):
            a, b = flat(i, j), flat(i + 1, j)
            dx, dy = get_shift(to_gs(frames[a]), to_gs(frames[b]))
            shift_down[(i, j)] = (dx, dy)

    # Cumulative positions: pos[i][j] = (x, y) top-left of frame (i,j) on a canvas
    pos = [[(0.0, 0.0)] * ny for _ in range(nx)]
    for i in range(nx):
        for j in range(ny):
            if i == 0 and j == 0:
                continue
            if j == 0:
                dx, dy = shift_down[(i - 1, 0)]
                pos[i][j] = (pos[i - 1][0][0] + dx, pos[i - 1][0][1] + dy)
            else:
                dx, dy = shift_right[(i, j - 1)]
                pos[i][j] = (pos[i][j - 1][0] + dx, pos[i][j - 1][1] + dy)

    # Canvas bounds (in pixel coords)
    min_x = min(pos[i][j][0] for i in range(nx) for j in range(ny))
    min_y = min(pos[i][j][1] for i in range(nx) for j in range(ny))
    max_x = max(pos[i][j][0] + W for i in range(nx) for j in range(ny))
    max_y = max(pos[i][j][1] + H for i in range(nx) for j in range(ny))
    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    # Accumulate with blending: sum and count per pixel
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    cnt = np.zeros((canvas_h, canvas_w), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            px = int(np.round(pos[i][j][0] - min_x))
            py = int(np.round(pos[i][j][1] - min_y))
            f = frames[flat(i, j)]
            # Clip to canvas
            x0, x1 = max(0, px), min(canvas_w, px + W)
            y0, y1 = max(0, py), min(canvas_h, py + H)
            fx0, fx1 = x0 - px, x1 - px
            fy0, fy1 = y0 - py, y1 - py
            if fx1 <= fx0 or fy1 <= fy0:
                continue
            acc[y0:y1, x0:x1, :] += f[fy0:fy1, fx0:fx1, :].astype(np.float64)
            cnt[y0:y1, x0:x1] += 1.0

    cnt = np.maximum(cnt, 1e-6)
    out = (acc / cnt[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    return out


def main():
    # Use filtered argv for argparse (Isaac args already in sys.argv for og)
    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a not in _isaac_args]
    parser = argparse.ArgumentParser(
        description="Visualize scene USD file and save PNG snapshot"
    )
    parser.add_argument(
        "usd_path",
        type=str,
        nargs="?",
        default=os.path.join(OUTPUT_FRAMES, "Rs_int_scene.usd"),
        help="Path to scene USD file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: output_frames/<usd_basename>_snapshot.png)",
    )
    parser.add_argument(
        "--camera-height",
        type=float,
        default=2.0,
        help="Camera height above floor (along scene up-axis)",
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        choices=("y", "z"),
        default="z",
        help="Scene up-axis: 'y' = BEHAVIOR/OmniGibson (floor in XZ, top-down = look -Y); 'z' = Isaac world (floor in XY, look -Z). Default: y",
    )
    parser.add_argument(
        "--view-angle",
        type=float,
        default=0.0,
        metavar="DEG",
        help="Tilt of camera from straight top-down, in degrees (0=straight down, 45=45° slanted view). Default: 0",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("W", "H"),
        help="Output resolution (default: 1280 720)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Output a video by sweeping camera x,y over the layout bounds (requires layout JSON for range).",
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Compose all swept frames into one image by estimating pixel overlap between adjacent frames (same sweep as --video, outputs one PNG).",
    )
    parser.add_argument(
        "--layout-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to scene layout JSON (state with object_registry). If not set, inferred from USD path and OMNIGIBSON_DATA_PATH.",
    )
    parser.add_argument(
        "--video-nx",
        type=int,
        default=20,
        metavar="N",
        help="Number of camera x positions in the sweep (default: 20).",
    )
    parser.add_argument(
        "--video-ny",
        type=int,
        default=20,
        metavar="N",
        help="Number of camera y positions in the sweep (default: 20).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        metavar="M",
        help="Margin (meters) added to layout bounds from JSON (default: 1.0).",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=10.0,
        metavar="FPS",
        help="Frames per second for output video (default: 10.0).",
    )
    args = parser.parse_args()
    sys.argv = _saved_argv  # Restore for OmniGibson

    usd_path = os.path.expanduser(args.usd_path)
    if not os.path.isfile(usd_path):
        print(f"Error: USD file not found: {usd_path}", file=sys.stderr)
        return 1

    base = os.path.splitext(os.path.basename(usd_path))[0]
    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        if args.compose:
            output_path = os.path.join(OUTPUT_FRAMES, f"{base}_compose.png")
        elif args.video:
            output_path = os.path.join(OUTPUT_FRAMES, f"{base}_sweep.mp4")
        else:
            output_path = os.path.join(OUTPUT_FRAMES, f"{base}_snapshot.png")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resolve layout JSON for video/compose mode (camera sweep bounds)
    layout_json_path = None
    if args.video or args.compose:
        if args.layout_json:
            layout_json_path = os.path.expanduser(args.layout_json)
            if not os.path.isfile(layout_json_path):
                print(f"Error: Layout JSON not found: {layout_json_path}", file=sys.stderr)
                return 1
        else:
            behavior_path = os.environ.get("OMNIGIBSON_DATA_PATH", _DEFAULT_BEHAVIOR_DATA_PATH)
            layout_json_path = _infer_layout_json_path(usd_path, behavior_path)
            if not layout_json_path:
                print("Error: --video/--compose requires layout JSON; pass --layout-json or set OMNIGIBSON_DATA_PATH so layout can be inferred.", file=sys.stderr)
                return 1
        print(f"Layout JSON: {layout_json_path}")

    print(f"Headless: {os.environ.get('OMNIGIBSON_HEADLESS', '1')}")
    print(f"RTX disabled: {os.environ.get('OMNI_DISABLE_RTX', '1')}")
    print(f"USD: {usd_path}")
    print(f"Output: {output_path}")
    print(f"Up-axis: {args.up_axis}, view-angle: {args.view_angle}° (0=straight down, 45=slanted)")

    try:
        import omnigibson as og
        import omnigibson.lazy as lazy
        import omnigibson.utils.transform_utils as T
        import numpy as np
        import torch as th
    except ImportError as e:
        print(f"Error: OmniGibson not available: {e}", file=sys.stderr)
        return 1

    try:
        from omnigibson.utils.usd_utils import add_asset_to_stage
    except ImportError:
        add_asset_to_stage = None

    try:
        # Minimal environment (no scene, no robots)
        cfg = {
            "scene": {
                "type": "Scene",
                "use_floor_plane": True,
                "use_skybox": False,
            },
            "robots": [],
            "render": {
                "viewer_width": args.resolution[0],
                "viewer_height": args.resolution[1],
            },
        }
        env = og.Environment(configs=cfg)
        print("Environment created")

        # Disable RTX in renderer
        try:
            carb_settings = lazy.carb.settings.get_settings()
            # carb_settings.set_bool("/rtx/enabled", False)
            # carb_settings.set_bool("/rtx/verifyDriverVersion/enabled", False)
            # print("RTX disabled in renderer")
        except Exception as e:
            print(f"Note: Could not disable RTX in settings: {e}")

        stage = og.sim.stage

        # Load scene USD at /World/scene_0 (same as OmniGibson uses)
        scene_prim_path = "/World/scene_0"
        print(f"Loading USD at {scene_prim_path}...")
        if add_asset_to_stage:
            add_asset_to_stage(asset_path=usd_path, prim_path=scene_prim_path)
        else:
            lazy.isaacsim.core.utils.stage.add_reference_to_stage(
                usd_path=usd_path,
                prim_path=scene_prim_path,
            )
        print("USD loaded")

        # Render to settle
        for _ in range(15):
            og.sim.render()

        # Camera: top-down or slanted view (above floor, looking down)
        camera = og.sim.viewer_camera
        tilt_rad = np.radians(args.view_angle)

        def set_camera_pose(cam_x, cam_y):
            if args.up_axis == "y":
                # Y-up: floor is XZ. Camera at (cam_x, height, cam_y) so we sweep X and Z.
                pos = th.tensor([float(cam_x), args.camera_height, float(cam_y)])
                q_td_x, q_td_w = -0.70710678, 0.70710678
                s, c = np.sin(tilt_rad / 2), np.cos(tilt_rad / 2)
                qw = c * q_td_w - s * q_td_x
                qx = c * q_td_x + s * q_td_w
                orn = th.tensor([float(qx), 0.0, 0.0, float(qw)])
            else:
                # Z-up: camera at (cam_x, cam_y, height).
                pos = th.tensor([float(cam_x), float(cam_y), args.camera_height])
                if args.view_angle == 0:
                    orn = th.tensor([0.0, 0.0, 0.0, 1.0])
                else:
                    s, c = np.sin(tilt_rad / 2), np.cos(tilt_rad / 2)
                    orn = th.tensor([float(s), 0.0, 0.0, float(c)])
            camera.set_position_orientation(position=pos, orientation=orn)

        def capture_frame():
            obs, _ = camera.get_obs()
            if isinstance(obs, dict) and "rgb" in obs:
                img = obs["rgb"]
                if hasattr(img, "cpu"):
                    img_np = img.cpu().numpy()
                else:
                    img_np = np.asarray(img)
                if img_np.ndim == 3 and img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                return img_np.astype(np.uint8)
            return None

        if args.video or args.compose:
            x_min, x_max, y_min, y_max = _get_layout_bounds_from_json(layout_json_path, margin=args.margin)
            print(f"Camera sweep bounds: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
            nx, ny = max(1, args.video_nx), max(1, args.video_ny)
            x_vals = np.linspace(x_min, x_max, nx)
            y_vals = np.linspace(y_min, y_max, ny)
            frames = []
            total = nx * ny
            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    set_camera_pose(x, y)
                    for _ in range(5):
                        og.sim.render()
                    frame = capture_frame()
                    if frame is None:
                        print("Error: Could not get RGB from camera", file=sys.stderr)
                        env.close()
                        return 1
                    frames.append(frame)
                    idx = i * ny + j + 1
                    if idx % 20 == 0 or idx == total:
                        print(f"  Frame {idx}/{total}")
            if not frames:
                print("Error: No frames captured", file=sys.stderr)
                env.close()
                return 1
            if args.compose:
                print("Composing frames (phase correlation for overlap)...")
                try:
                    composed = _compose_frames_into_image(frames, nx, ny)
                    try:
                        from PIL import Image
                        Image.fromarray(composed).save(output_path)
                    except ImportError:
                        import cv2
                        cv2.imwrite(output_path, cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
                    print(f"Saved composed image: {output_path} ({len(frames)} frames)")
                except Exception as e:
                    print(f"Error composing frames: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()
                    env.close()
                    return 1
            else:
                try:
                    import imageio
                    imageio.mimsave(output_path, frames, fps=args.video_fps)
                    print(f"Saved video: {output_path} ({len(frames)} frames)")
                except ImportError:
                    print("Error: imageio not installed; cannot save video. Install with: pip install imageio", file=sys.stderr)
                    env.close()
                    return 1
                except Exception as e:
                    print(f"Error writing video: {e}", file=sys.stderr)
                    env.close()
                    return 1
        else:
            # Single snapshot: use center of layout if we have JSON, else fixed position
            if args.layout_json and os.path.isfile(os.path.expanduser(args.layout_json)):
                x_min, x_max, y_min, y_max = _get_layout_bounds_from_json(os.path.expanduser(args.layout_json), margin=0)
                cam_x = (x_min + x_max) / 2
                cam_y = (y_min + y_max) / 2
            else:
                cam_x, cam_y = 0.0, 0.0
            set_camera_pose(cam_x, cam_y)
            for _ in range(10):
                og.sim.render()
            img_np = capture_frame()
            if img_np is None:
                print("Error: Could not get RGB from camera", file=sys.stderr)
                env.close()
                return 1
            try:
                from PIL import Image
                Image.fromarray(img_np).save(output_path)
                print(f"Saved: {output_path}")
            except ImportError:
                import cv2
                cv2.imwrite(output_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                print(f"Saved: {output_path}")

        env.close()
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
