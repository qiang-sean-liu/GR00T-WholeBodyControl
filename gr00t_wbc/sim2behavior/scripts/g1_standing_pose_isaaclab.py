"""
G1 standing pose matching Isaac Lab Arena G1SceneCfg.init_state.joint_pos.

Used by run_omnigibson_g1_with_gr00t.py so the robot starts in the same stable
standing pose as in isaaclab_arena (e.g. galileo_g1_locomanip_pick_and_place).
No dependency on isaaclab or isaaclab_arena.

Source: isaaclab_arena/embodiments/g1/g1.py G1SceneCfg.robot.init_state.joint_pos.
Joints not listed there (wrist, hand) default to 0.0 for a neutral stand.
"""

# Full 43-DOF standing pose by joint name [rad].
# Matches isaaclab_arena G1SceneCfg.init_state.joint_pos; wrist/hand set to 0.
G1_STANDING_JOINT_POS_BY_NAME = {
    # Legs + waist (same as G1SceneCfg and g1_gear_wbc default_angles)
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # Arms (G1SceneCfg lists shoulder + elbow only; wrist/hand 0)
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "left_hand_index_0_joint": 0.0,
    "left_hand_index_1_joint": 0.0,
    "left_hand_middle_0_joint": 0.0,
    "left_hand_middle_1_joint": 0.0,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": 0.0,
    "left_hand_thumb_2_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "right_hand_index_0_joint": 0.0,
    "right_hand_index_1_joint": 0.0,
    "right_hand_middle_0_joint": 0.0,
    "right_hand_middle_1_joint": 0.0,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": 0.0,
    "right_hand_thumb_2_joint": 0.0,
}


def build_standing_pinocchio_from_names(robot_model, n_joints: int = 43):
    """
    Build standing pose vector in Pinocchio/WBC order from G1_STANDING_JOINT_POS_BY_NAME.

    Uses robot_model.dof_index(joint_name) to place each value. Joints not in
    the dict or not in robot_model remain 0. No dependency on Isaac Lab or yaml path.

    Args:
        robot_model: Model with dof_index(joint_name) and joint_names.
        n_joints: Length of returned array (default 43).

    Returns:
        np.ndarray of shape (n_joints,) in Pinocchio order.
    """
    import numpy as np

    out = np.zeros(n_joints, dtype=np.float64)
    for jname, value in G1_STANDING_JOINT_POS_BY_NAME.items():
        try:
            pidx = robot_model.dof_index(jname)
        except (ValueError, KeyError, TypeError):
            continue
        if 0 <= pidx < n_joints:
            out[pidx] = value
    return out
