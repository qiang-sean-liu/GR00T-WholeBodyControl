#!/usr/bin/env python3
"""
Patch the Unitree G1 USD so the D435 camera extrinsics match MuJoCo's robot0_oak_egoview.

Target (in torso_link frame): pos = [0.10209156, -0.00937542, 0.42446595] m,
  quat = [0.64367383, 0.26523914, -0.27106013, -0.66472446] (w,x,y,z), fovy = 79.5°.

In the USD, the Camera is under d435_link. We set the Camera's pose *in d435_link frame* so
the combined pose (d435_link in torso × Camera in d435_link) equals the MuJoCo ego camera.

Computed translate (m) and orient (USD quat x,y,z,w) for Camera in d435_link:
  translate = (0.033976, -0.026905, 0.029194)
  orient    = raw match then 180° roll: raw (0.51..., -0.51..., -0.50..., 0.48...) gave upside-down
  image (MuJoCo vs USD image-Y convention). Final orient = (oy, -ox, ow, -oz) of raw
  -> (-0.50776, -0.51093, 0.47955, 0.50116) so the view is right-side up.

Usage:
  python patch_g1_usd_d435_oak_egoview.py [path_to_unitree_g1.usda]
  If no path is given, uses OMNIGIBSON_DATA_PATH or BEHAVIOR_DATA_PATH + omnigibson-robot-assets/.../unitree_g1.usda
"""
from __future__ import annotations

import os
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path

# MuJoCo oak_egoview-matching pose for D435 Camera *in d435_link frame*
D435_TRANSLATE_OAK_EGOVIEW = (0.033976, -0.026905, 0.029194)  # m, (x,y,z)
# USD quat is (x, y, z, w). Computed orient to match MuJoCo torso pose:
_ORIENT_OAK_EGOVIEW_RAW = (0.510929544, -0.5077606323, -0.5011600631, 0.4795505526)
# The raw orient makes the rendered image upside down (MuJoCo vs USD/Isaac image-Y convention).
# Apply 180° roll (rotation around camera view axis): orient * q_z180, q_z180 = (0,0,1,0) in (x,y,z,w).
# Quat mult (w,x,y,z): (ow,ox,oy,oz)*(0,0,0,1) -> (-oz, oy, -ox, ow) -> (x,y,z,w) = (oy, -ox, ow, -oz).
def _apply_roll_180_xydz(ox: float, oy: float, oz: float, ow: float) -> tuple[float, float, float, float]:
    return (oy, -ox, ow, -oz)
D435_ORIENT_OAK_EGOVIEW = _apply_roll_180_xydz(*_ORIENT_OAK_EGOVIEW_RAW)

TRANSLATE_LINE = f'double3 xformOp:translate = ({D435_TRANSLATE_OAK_EGOVIEW[0]}, {D435_TRANSLATE_OAK_EGOVIEW[1]}, {D435_TRANSLATE_OAK_EGOVIEW[2]})'
ORIENT_LINE = f'quatd xformOp:orient = ({D435_ORIENT_OAK_EGOVIEW[0]}, {D435_ORIENT_OAK_EGOVIEW[1]}, {D435_ORIENT_OAK_EGOVIEW[2]}, {D435_ORIENT_OAK_EGOVIEW[3]})'


def resolve_usd_path() -> Path | None:
    """Resolve path to unitree_g1.usda from env or default locations."""
    rel = Path("omnigibson-robot-assets") / "models" / "unitree_g1" / "usd" / "unitree_g1.usda"
    for env in ("OMNIGIBSON_DATA_PATH", "BEHAVIOR_DATA_PATH"):
        base = os.environ.get(env)
        if base:
            p = Path(base) / rel
            if p.exists():
                return p
    # Try BEHAVIOR-1K datasets relative to common roots
    for root in (Path(__file__).resolve().parents[5], Path.home()):
        for sub in ("BEHAVIOR-1K/datasets", "datasets"):
            p = root / sub / rel
            if p.exists():
                return p
    return None


def find_d435_camera_prim(stage):
    """Return the Camera prim under d435_link, or None."""
    try:
        from pxr import Usd, UsdGeom
    except ImportError:
        return None
    for prim in stage.Traverse():
        if prim.GetName() == "Camera":
            parent = prim.GetParent()
            if parent and parent.GetName() == "d435_link":
                return prim
    return None


def patch_usd_text(usd_path: Path, backup: bool = True) -> bool:
    """Edit .usda as text when pxr is not available. Only works for .usda."""
    if usd_path.suffix.lower() != ".usda":
        return False
    text = usd_path.read_text(encoding="utf-8", errors="replace")
    # Find the Camera block under d435_link: "def Xform \"d435_link\"" then "def Camera \"Camera\" {".
    d435_m = re.search(r'def Xform\s+"d435_link"\s*\{', text)
    if not d435_m:
        print("Could not find def Xform \"d435_link\" in the USD.", file=sys.stderr)
        return False
    after_d435 = text[d435_m.end() :]
    cam_m = re.search(r'def Camera\s+"Camera"\s*\{', after_d435)
    if not cam_m:
        print("Could not find def Camera \"Camera\" under d435_link.", file=sys.stderr)
        return False
    block_start = d435_m.end() + cam_m.end()
    block = after_d435[cam_m.end() :]
    # End of Camera block: match braces so we don't include nested content
    depth = 1
    i = 0
    while i < len(block) and depth > 0:
        if block[i] == "{":
            depth += 1
        elif block[i] == "}":
            depth -= 1
        i += 1
    block_end = block_start + i - (1 if depth == 0 else 0)
    camera_block = text[block_start:block_end]
    translate_pat = re.compile(r"double3 xformOp:translate\s*=\s*\([^)]+\)")
    orient_pat = re.compile(r"quatd xformOp:orient\s*=\s*\([^)]+\)")
    new_block = translate_pat.sub(TRANSLATE_LINE, camera_block, count=1)
    new_block = orient_pat.sub(ORIENT_LINE, new_block, count=1)
    if new_block == camera_block:
        print("No translate or orient lines found in d435_link Camera block.", file=sys.stderr)
        return False
    new_text = text[:block_start] + new_block + text[block_end:]
    if backup:
        backup_path = usd_path.with_suffix(usd_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(usd_path, backup_path)
        print(f"Backed up USD to {backup_path}")
    usd_path.write_text(new_text, encoding="utf-8")
    print(f"Patched D435 Camera (text fallback): translate={D435_TRANSLATE_OAK_EGOVIEW}, orient (x,y,z,w)={D435_ORIENT_OAK_EGOVIEW}")
    return True


def patch_usd(usd_path: Path, backup: bool = True) -> bool:
    """Set D435 Camera translate and orient in the USD. Returns True on success."""
    try:
        from pxr import Usd, UsdGeom, Gf
    except ImportError:
        if usd_path.suffix.lower() in (".usda",):
            print("pxr (Usd) not available; using text-based patch for .usda.", file=sys.stderr)
            return patch_usd_text(usd_path, backup=backup)
        print("pxr (Usd) not available. Install Omniverse USD or use a .usda file for text fallback.", file=sys.stderr)
        return False

    path_str = str(usd_path.resolve())
    stage = Usd.Stage.Open(path_str)
    if not stage:
        print(f"Failed to open stage: {path_str}", file=sys.stderr)
        return False

    cam_prim = find_d435_camera_prim(stage)
    if not cam_prim:
        print("No Camera prim under d435_link found in the USD.", file=sys.stderr)
        return False

    if backup:
        backup_path = usd_path.with_suffix(usd_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        shutil.copy2(usd_path, backup_path)
        print(f"Backed up USD to {backup_path}")

    xform = UsdGeom.Xformable(cam_prim)
    # Get or create translate/orient ops (USD applies in reverse order; we want translate then orient)
    ops = xform.GetOrderedXformOps()
    translate_op = None
    orient_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if not translate_op:
        translate_op = xform.AddTranslateOp()
    if not orient_op:
        orient_op = xform.AddOrientOp()

    translate_op.Set(Gf.Vec3d(*D435_TRANSLATE_OAK_EGOVIEW))
    # Gf.Quatd(real, i, j, k) = (w, x, y, z); we have (x,y,z,w)
    x, y, z, w = D435_ORIENT_OAK_EGOVIEW
    orient_op.Set(Gf.Quatd(w, x, y, z))

    stage.Save()
    print(f"Patched D435 Camera at {cam_prim.GetPath()}: translate={D435_TRANSLATE_OAK_EGOVIEW}, orient (x,y,z,w)={D435_ORIENT_OAK_EGOVIEW}")
    return True


def main():
    if len(sys.argv) >= 2:
        usd_path = Path(sys.argv[1])
        if not usd_path.exists():
            print(f"File not found: {usd_path}", file=sys.stderr)
            sys.exit(1)
    else:
        usd_path = resolve_usd_path()
        if not usd_path:
            print(
                "Unitree G1 USD not found. Set OMNIGIBSON_DATA_PATH or BEHAVIOR_DATA_PATH, or pass the path:\n"
                "  python patch_g1_usd_d435_oak_egoview.py <path/to/unitree_g1.usda>",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Using USD: {usd_path}")

    ok = patch_usd(usd_path, backup=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
