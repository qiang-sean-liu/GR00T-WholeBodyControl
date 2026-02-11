# Robot Model Relationships: Understanding Multiple Robot Definitions

This document explains what `instantiate_g1_robot_model()` does and how it relates to other robot definitions in the codebase.

## Overview: Three Layers of Robot Definition

There are **three separate but related** robot definitions:

1. **Simulator-level robot** (OmniGibson/MuJoCo) - Physical robot in simulation
2. **RobotModel** (gr00t_wbc) - Abstract kinematic/dynamic model for control
3. **Policy-level robot** (GR00T) - Robot type tag for policy selection

## 1. `instantiate_g1_robot_model()` - What It Does

**Location:** `gr00t_wbc/control/robot_model/instantiation/g1.py`

**Purpose:** Creates a `RobotModel` instance that represents the G1 robot's **kinematic and dynamic properties** for control/planning purposes.

**What it does:**
```python
def instantiate_g1_robot_model(
    waist_location: Literal["lower_body", "upper_body", "lower_and_upper_body"] = "lower_body",
    high_elbow_pose: bool = False,
):
```

1. **Finds URDF path:** Calculates path to `g1_29dof_with_hand.urdf` (5 levels up from script)
2. **Creates RobotModel:** Instantiates `RobotModel` class with:
   - URDF file path
   - Asset/mesh directory path
   - `G1SupplementalInfo` (joint groups, limits, hand frame names, etc.)
3. **Configures waist location:** Determines how waist joints are grouped (lower_body vs upper_body)
4. **Returns:** A `RobotModel` object that provides:
   - Joint indices and names
   - Forward/inverse kinematics (via Pinocchio)
   - Joint groups (left_arm, right_arm, waist, left_leg, right_leg, left_hand, right_hand)
   - Joint limits
   - End-effector frame names

**Key point:** This is **NOT** the simulator robot - it's an abstract model used for:
- Converting between joint orders (actuated vs full configuration)
- Computing forward kinematics (wrist poses from joint angles)
- Grouping joints for control (upper body, lower body, etc.)
- Providing joint limits and safety checks

## 2. Where `instantiate_g1_robot_model()` is Called

### Direct calls:
- Tests (`tests/control/robot_model/robot_model_test.py`)
- Visualization (`control/visualization/humanoid_visualizer.py`)
- Teleop scripts (`control/main/teleop/run_teleop_policy_loop.py`)

### Via `get_robot_type_and_model()` wrapper:
- **SyncEnv** (MuJoCo/RoboCasa): `sync_env.py:37` - Creates RobotModel for observation/action conversion
- **OmniGibson adapter script**: `run_omnigibson_g1_with_gr00t.py:245` - Creates RobotModel for adapters
- **Data collection**: `sync_sim_utils.py` - Creates RobotModel for data export
- **Playback**: `playback_sync_sim_data.py` - Creates RobotModel for replay

## 3. Relationship Between Robot Definitions

### Layer 1: Simulator Robot (Physical Representation)

**OmniGibson:**
- **Config:** `configs/g1_standalone.yaml`
- **Type:** `"UnitreeG1"` (OmniGibson built-in)
- **Name:** `"unitree_g1"`
- **URDF:** Loaded from `omnigibson-robot-assets/models/unitree_g1/` (USD format)
- **Purpose:** Physical robot in OmniGibson simulation
- **Controls:** OmniGibson controllers (IK, joint controllers, etc.)

**MuJoCo/RoboCasa:**
- **Config:** RoboCasa environment configs
- **Type:** `"G1"` or `"g1"` (pattern-matched)
- **Name:** Passed to `SyncEnv.__init__(robot_name="g1")`
- **URDF:** MuJoCo XML (converted from URDF)
- **Purpose:** Physical robot in MuJoCo simulation
- **Controls:** RoboCasa controllers (WBC, IK, etc.)

### Layer 2: RobotModel (Abstract Control Model)

**Created by:** `instantiate_g1_robot_model()` → `RobotModel` class

**URDF:** `gr00t_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand.urdf`

**Purpose:**
- **Observation conversion:** Maps simulator joint order → policy joint order
- **Action conversion:** Maps policy joint order → simulator joint order
- **Kinematics:** Computes end-effector poses from joint angles
- **Joint grouping:** Provides indices for `state.left_arm`, `state.right_arm`, etc.
- **Safety:** Joint limits and safety checks

**Key insight:** The RobotModel URDF may differ from simulator URDFs:
- Different joint orderings
- Different DOF representations
- Different frame names
- But represents the **same physical robot**

### Layer 3: Policy Robot Tag (GR00T)

**Tag:** `EmbodimentTag.UNITREE_G1` or `"UNITREE_G1"`

**Purpose:** Tells GR00T policy which robot-specific processing to use:
- Observation normalization
- Action scaling
- Camera selection
- Joint group handling

**Used in:**
- `run_gr00t_server.py --embodiment-tag UNITREE_G1`
- Policy loading: `Gr00tPolicy.from_pretrained(..., embodiment_tag=EmbodimentTag.UNITREE_G1)`

## 4. How They Work Together

### Example: OmniGibson + GR00T Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. OmniGibson Environment                                    │
│    - Loads "UnitreeG1" robot from USD                       │
│    - Provides obs["unitree_g1"]["proprio"]["qpos"]          │
│    - Expects action["unitree_g1"]["qpos"]                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. OmniGibsonToGR00TObservationAdapter                      │
│    - Uses RobotModel (from instantiate_g1_robot_model)      │
│    - Converts obs → GR00T format                            │
│    - Maps camera keys                                        │
│    - Adds state.* joint groups                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. GR00T Policy (UNITREE_G1)                                │
│    - Processes observation                                   │
│    - Outputs action["q"] in Pinocchio joint order           │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. GR00TToOmniGibsonActionAdapter                            │
│    - Uses RobotModel (same instance)                        │
│    - Converts action["q"] → OmniGibson format                │
│    - Handles joint order conversion                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. OmniGibson Environment                                    │
│    - Applies action to physical robot                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Relationships:

1. **Simulator robot ≠ RobotModel:**
   - Simulator has its own URDF/USD (may differ in joint order, DOF, etc.)
   - RobotModel is a standardized representation for control

2. **RobotModel is shared:**
   - Same `RobotModel` instance used by observation and action adapters
   - Provides consistent joint ordering and grouping

3. **Policy tag is independent:**
   - `UNITREE_G1` tag tells policy how to process observations/actions
   - Doesn't directly reference RobotModel or simulator

## 5. Why Multiple Definitions?

**Separation of concerns:**
- **Simulator:** Handles physics, rendering, collision detection
- **RobotModel:** Handles kinematics, joint ordering, control abstractions
- **Policy:** Handles high-level behavior, doesn't need low-level robot details

**Flexibility:**
- Different simulators can use different URDFs/USDs
- RobotModel provides a consistent interface for control
- Policy works with abstract joint groups, not simulator-specific details

**Example:** MuJoCo might use a simplified URDF, OmniGibson uses USD with cameras, but both use the same `RobotModel` for control.

## 6. Common Confusion Points

**Q: Why does OmniGibson config use `"UnitreeG1"` but RobotModel uses `"G1"`?**

**A:** They're different systems:
- `"UnitreeG1"` is OmniGibson's registered robot type (in `REGISTERED_ROBOTS`)
- `"G1"` is the pattern-matched name for `get_robot_type_and_model()` (checks `startswith("g1")`)

**Q: Can I use the same URDF for both?**

**A:** Not necessarily. Simulators may need:
- Different joint orderings
- Different DOF representations (e.g., floating base vs fixed)
- Different mesh formats (USD vs URDF)
- Different controller configurations

**Q: Why does `instantiate_g1_robot_model()` hardcode the URDF path?**

**A:** It ensures consistency - all control code uses the same URDF, regardless of which simulator is running. The path is calculated relative to the script location to work from any directory.

## Summary

- **`instantiate_g1_robot_model()`** creates an abstract `RobotModel` for control/kinematics
- **Simulator robots** (OmniGibson/MuJoCo) are separate physical representations
- **Policy tag** (`UNITREE_G1`) is for policy-level robot-specific handling
- **They work together** via adapters that convert between formats using the shared `RobotModel`
