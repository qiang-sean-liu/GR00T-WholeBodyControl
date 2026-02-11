#!/usr/bin/env python3
"""Create meshes symlink in sim2behavior/resources/robots/g1 pointing to sim2mujoco meshes."""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM2BEHAVIOR_ROOT = os.path.dirname(SCRIPT_DIR)
G1_DIR = os.path.join(SIM2BEHAVIOR_ROOT, "resources", "robots", "g1")
# sim2mujoco is sibling of sim2behavior under gr00t_wbc
SIM2MUJOCO_MESHES = os.path.join(SIM2BEHAVIOR_ROOT, "..", "sim2mujoco", "resources", "robots", "g1", "meshes")
LINK_PATH = os.path.join(G1_DIR, "meshes")

def main():
    sim2mujoco_abs = os.path.normpath(os.path.abspath(SIM2MUJOCO_MESHES))
    if not os.path.isdir(sim2mujoco_abs):
        print(f"sim2mujoco meshes not found: {sim2mujoco_abs}")
        return 1
    if os.path.islink(LINK_PATH):
        print(f"Already a symlink: {LINK_PATH}")
        return 0
    if os.path.isdir(LINK_PATH):
        print(f"meshes already exists as directory: {LINK_PATH}")
        return 0
    try:
        os.symlink(sim2mujoco_abs, LINK_PATH)
        print(f"Created symlink: {LINK_PATH} -> {sim2mujoco_abs}")
    except OSError as e:
        print(f"Failed to create symlink: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
