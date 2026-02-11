#!/usr/bin/env python3
"""
Validate a saved scene USD file (e.g. from OMNIGIBSON_SAVE_SCENE_USD_TO).

Run with OmniGibson/Isaac Sim Python (which has pxr):
  python scripts/validate_scene_usd.py /path/to/scene.usd

Or with system Python for basic format check only:
  python scripts/validate_scene_usd.py /path/to/scene.usd
"""
import os
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_scene_usd.py <path/to/scene.usd>", file=sys.stderr)
        sys.exit(1)
    path = os.path.expanduser(sys.argv[1])
    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    # Basic format check (no pxr required)
    with open(path, "rb") as f:
        header = f.read(12)
    if header[:8] != b"PXR-USDC":
        print("FAIL: Not a valid USD crate file (missing PXR-USDC magic)")
        sys.exit(1)
    print("Format: USD crate OK")

    # Full validation with pxr (requires Isaac Sim / OmniGibson env)
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(path)
        if not stage:
            print("FAIL: Usd.Stage.Open returned None")
            sys.exit(1)
        print("Stage: opened successfully")

        world = stage.GetPrimAtPath("/World")
        if not world:
            print("WARN: /World prim not found (expected for OmniGibson prebuilt scene)")
        else:
            children = list(world.GetChildren())
            print(f"World children: {len(children)} prims")

        count = sum(1 for _ in stage.Traverse())
        print(f"Total prims: {count}")

        default_prim = stage.GetDefaultPrim()
        if default_prim:
            print(f"Default prim: {default_prim.GetPath().pathString}")
        print("Validation: PASSED")
    except ImportError:
        print("(Full validation skipped: pxr not available. Run with OmniGibson/Isaac Sim Python for full check.)")


if __name__ == "__main__":
    main()
