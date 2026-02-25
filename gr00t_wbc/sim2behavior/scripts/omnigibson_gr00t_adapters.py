"""
Observation and action adapters for running GR00T policy with OmniGibson G1 robot.

Physical embodiment: g1_29dof_with_hand_rev_1_0.usd (same as Isaac Lab Arena) when available;
otherwise unitree_g1.usda. All conversions between simulator and GR00T are by joint *name*,
so articulation order from the loaded USD does not matter.

Maps between:
- OmniGibson observation format (obs["unitree_g1"], obs["external"], etc.)
- GR00T policy observation format (q, dq, camera images, state.*, annotation.human.task_description)

And:
- GR00T policy action format ({"q": ...} in Pinocchio joint order)
- OmniGibson action format (dict with robot name keys, joint targets)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Tuple
import cv2


class OmniGibsonToGR00TObservationAdapter:
    """Convert OmniGibson observations to GR00T policy observation format."""

    def __init__(
        self,
        robot_model,
        robot_name: str = "unitree_g1",
        camera_mapping: Optional[Dict[str, str]] = None,
        task_description: str = "",
        video_time_horizon: int = 1,
        state_time_horizon: int = 1,
    ):
        """
        Args:
            robot_model: RobotModel instance (from gr00t_wbc.control.robot_model)
            robot_name: Name of the robot in OmniGibson obs dict (default: "unitree_g1")
            camera_mapping: Dict mapping OmniGibson camera keys to GR00T camera keys.
                Default: {"unitree_g1:d435_link:Camera:0": "robot0_oak_egoview",
                          "unitree_g1:mid360_link:Camera:0": "robot0_rs_tppview"}
            task_description: Language instruction for the task (default: empty string)
            video_time_horizon: Time dimension T for video keys (B, T, H, W, C). Default 1.
                Should match policy's modality_configs["video"].delta_indices length.
            state_time_horizon: Time dimension T for state keys (B, T, D). Default 1.
                Should match policy's modality_configs["state"].delta_indices length.
        """
        self.robot_model = robot_model
        self.robot_name = robot_name
        self.task_description = task_description
        self.video_time_horizon = video_time_horizon
        self.state_time_horizon = state_time_horizon

        # Default camera mapping: OmniGibson camera keys → RoboCasa camera keys
        # head_camera (Arena-aligned, head_link) is preferred for ego_view when present; else d435.
        if camera_mapping is None:
            self.camera_mapping = {
                "unitree_g1:d435_link:Camera:0": "robot0_oak_egoview",
                "unitree_g1:mid360_link:Camera:0": "robot0_rs_tppview",
            }
            # Order of sources for ego_view: head_camera (Arena) first, then d435
            self._ego_camera_priority = ["head_camera", "unitree_g1:d435_link:Camera:0"]
        else:
            self.camera_mapping = camera_mapping
            self._ego_camera_priority = ["head_camera", "unitree_g1:d435_link:Camera:0"]

        # Reverse mapping for lookup
        self.reverse_camera_mapping = {v: k for k, v in self.camera_mapping.items()}
        
        # Import CameraKeyMapper and RS_VIEW constants (same as isaaclab_arena_g1 / GR00T evaluation)
        try:
            from gr00t_wbc.control.envs.robocasa.utils.cam_key_converter import CameraKeyMapper
            from gr00t_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
            self.camera_key_mapper = CameraKeyMapper()
            self._camera_height = RS_VIEW_CAMERA_HEIGHT
            self._camera_width = RS_VIEW_CAMERA_WIDTH
        except ImportError:
            # Fallback: use direct mapping if CameraKeyMapper not available; use 480x640 to match constants
            self.camera_key_mapper = None
            self._camera_height = 480
            self._camera_width = 640

    def adapt(self, og_obs: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """
        Convert OmniGibson observation to GR00T policy observation format.

        Args:
            og_obs: OmniGibson observation dict (from env.reset() or env.step())
            verbose: If True, print debug info about observation keys

        Returns:
            Observation dict in GR00T policy format:
            {
                "q": np.ndarray,  # joint positions (num_joints,)
                "dq": np.ndarray,  # joint velocities
                "ddq": np.ndarray,  # joint accelerations (zeros if not available)
                "tau_est": np.ndarray,  # estimated torques (zeros if not available)
                "floating_base_pose": np.ndarray,  # (7,) [x,y,z,qx,qy,qz,qw]
                "floating_base_vel": np.ndarray,  # (6,) [vx,vy,vz,wx,wy,wz]
                "floating_base_acc": np.ndarray,  # (6,) zeros if not available
                "wrist_pose": np.ndarray,  # (14,) [left_wrist_7d, right_wrist_7d]
                "state.left_arm": np.ndarray,
                "state.right_arm": np.ndarray,
                "state.waist": np.ndarray,
                "state.left_leg": np.ndarray,
                "state.right_leg": np.ndarray,
                "state.left_hand": np.ndarray,
                "state.right_hand": np.ndarray,
                "robot0_oak_egoview_image": np.ndarray,  # (H, W, 3) uint8
                "robot0_rs_tppview_image": np.ndarray,
                "video.robot0_oak_egoview": np.ndarray,  # same as _image
                "video.robot0_rs_tppview": np.ndarray,
                "annotation.human.task_description": str,
            }
        """
        if verbose:
            print(f"OmniGibson obs top-level keys: {list(og_obs.keys())}")

        robot_obs = og_obs.get(self.robot_name, {})
        if verbose:
            print(f"Robot obs keys for '{self.robot_name}': {list(robot_obs.keys()) if isinstance(robot_obs, dict) else type(robot_obs)}")

        # Extract joint state.
        # IMPORTANT: Prefer robot_obs["joint_qpos"] (set by the run script in
        # Pinocchio order) over the raw OG proprio tensor.  OmniGibson's proprio
        # is a *flat* concatenation of sin/cos encodings, velocities, EEF poses,
        # etc.—NOT raw joint positions.  Taking its first N elements would give
        # sin(base_joints) instead of actual positions.
        q = None
        dq = None
        proprio = robot_obs.get("proprio", {}) if isinstance(robot_obs, dict) else {}

        if isinstance(robot_obs, dict):
            for key in ("joint_qpos", "qpos", "joint_positions", "q"):
                if key in robot_obs:
                    q = np.asarray(robot_obs[key])
                    break

        # Fall back to proprio only if no explicit joint data was provided
        if q is None or len(q) == 0:
            if verbose and proprio is not None:
                if isinstance(proprio, dict):
                    print(f"Proprio keys: {list(proprio.keys())}")
                else:
                    print(f"Proprio type: {type(proprio)}, shape: {getattr(proprio, 'shape', 'N/A')}")
            if isinstance(proprio, dict):
                for key in ("joint_qpos", "qpos", "joint_positions", "q"):
                    if key in proprio:
                        q = np.asarray(proprio[key])
                        break
                if q is None and "joints" in proprio and isinstance(proprio["joints"], dict):
                    q = np.asarray(proprio["joints"].get("positions", []))

        if q is None or len(q) == 0:
            if verbose:
                print(f"Warning: Could not find joint positions, using zeros. Available keys: {list(robot_obs.keys())}")
            q = np.zeros(self.robot_model.num_joints)
        elif len(q) != self.robot_model.num_joints:
            if verbose:
                print(f"Warning: Joint positions length {len(q)} != expected {self.robot_model.num_joints}, padding/truncating")
            if len(q) < self.robot_model.num_joints:
                q = np.pad(q, (0, self.robot_model.num_joints - len(q)))
            else:
                q = q[:self.robot_model.num_joints]

        # Extract velocities (same priority: robot_obs first, proprio fallback)
        if isinstance(robot_obs, dict):
            for key in ("joint_velocities", "qvel", "dq"):
                if key in robot_obs:
                    dq = np.asarray(robot_obs[key])
                    break

        if (dq is None or len(dq) == 0):
            if isinstance(proprio, dict):
                for key in ("joint_velocities", "qvel", "dq"):
                    if key in proprio:
                        dq = np.asarray(proprio[key])
                        break
                if dq is None and "joints" in proprio and isinstance(proprio["joints"], dict):
                    dq = np.asarray(proprio["joints"].get("velocities", []))

        if dq is None or len(dq) == 0:
            dq = np.zeros_like(q)
        elif len(dq) != len(q):
            if len(dq) < len(q):
                dq = np.pad(dq, (0, len(q) - len(dq)))
            else:
                dq = dq[:len(q)]

        # OmniGibson may not provide ddq/tau_est; use zeros
        ddq = np.zeros_like(q)
        tau_est = np.zeros_like(q)

        # Extract base pose/velocity (if available in proprio)
        # OmniGibson may store base pose in different keys; try multiple possibilities
        base_pos = None
        base_quat = None
        base_vel = None
        base_ang_vel = None

        # Only try dict access if proprio is actually a dict
        if isinstance(proprio, dict):
            # Try different key names
            for key in ["base_pos", "base_position", "position", "pos"]:
                if key in proprio:
                    base_pos = np.asarray(proprio[key])[:3]  # take first 3
                    break
            for key in ["base_quat", "base_quaternion", "quaternion", "quat", "orientation"]:
                if key in proprio:
                    quat = np.asarray(proprio[key])
                    if len(quat) >= 4:
                        base_quat = quat[:4]  # [x,y,z,w] or [w,x,y,z] - assume [x,y,z,w]
                        break
            for key in ["base_lin_vel", "base_linear_velocity", "linear_velocity", "lin_vel"]:
                if key in proprio:
                    base_vel = np.asarray(proprio[key])[:3]
                    break
            for key in ["base_ang_vel", "base_angular_velocity", "angular_velocity", "ang_vel"]:
                if key in proprio:
                    base_ang_vel = np.asarray(proprio[key])[:3]
                    break

        # Fallback: run script may add robot pose/vel to robot_obs (for WBC)
        if isinstance(robot_obs, dict):
            if base_pos is None and "robot_pos" in robot_obs:
                base_pos = np.asarray(robot_obs["robot_pos"])[:3]
            if base_quat is None and "robot_quat" in robot_obs:
                base_quat = np.asarray(robot_obs["robot_quat"])[:4]
            if base_vel is None and "robot_lin_vel" in robot_obs:
                base_vel = np.asarray(robot_obs["robot_lin_vel"])[:3]
            if base_ang_vel is None and "robot_ang_vel" in robot_obs:
                base_ang_vel = np.asarray(robot_obs["robot_ang_vel"])[:3]
        
        # Set defaults if not found
        if base_pos is None:
            base_pos = np.array([0.0, 0.0, 0.8])  # default spawn height
        if base_quat is None:
            base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        if base_vel is None:
            base_vel = np.array([0.0, 0.0, 0.0])
        if base_ang_vel is None:
            base_ang_vel = np.array([0.0, 0.0, 0.0])

        # WBC (gear_wbc_utils.get_gravity_orientation) expects quat as (w,x,y,z); OmniGibson returns (x,y,z,w).
        # Convert so floating_base_pose[3:7] is (w,x,y,z) for correct gravity direction and parity with isaaclab_arena_g1.
        base_quat = np.asarray(base_quat).ravel()[:4]
        if len(base_quat) == 4:
            quat_wxyz = np.array([float(base_quat[3]), float(base_quat[0]), float(base_quat[1]), float(base_quat[2])], dtype=np.float32)
        else:
            quat_wxyz = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        floating_base_pose = np.concatenate([np.asarray(base_pos).ravel()[:3], quat_wxyz]).astype(np.float32)  # (7,)
        floating_base_vel = np.concatenate([base_vel, base_ang_vel])  # (6,)
        floating_base_acc = np.zeros(6)  # not typically available

        # Compute wrist poses from joint configuration
        wrist_pose = self._compute_wrist_poses(q)

        # Build base observation dict (these are kept as-is, policy wrapper may handle them)
        # But state.* keys need (B, T, D) shape
        obs = {
            "q": q.astype(np.float32),
            "dq": dq.astype(np.float32),
            "ddq": ddq.astype(np.float32),
            "tau_est": tau_est.astype(np.float32),
            "floating_base_pose": floating_base_pose.astype(np.float32),
            "floating_base_vel": floating_base_vel.astype(np.float32),
            "floating_base_acc": floating_base_acc.astype(np.float32),
            "wrist_pose": wrist_pose.astype(np.float32),
        }

        # Add state.* keys (joint group slices) with batch and time dimensions (B, T, D)
        # Inline prepare_observation_for_eval to avoid importing n1_utils (which has heavy dependencies like onnxruntime)
        whole_q = np.asarray(obs["q"]).reshape(-1)  # ensure 1D for consistent slicing
        if whole_q.shape[0] == self.robot_model.num_joints:
            # Extract joint groups
            left_arm_q = whole_q[..., self.robot_model.get_joint_group_indices("left_arm")]
            right_arm_q = whole_q[..., self.robot_model.get_joint_group_indices("right_arm")]
            waist_q = whole_q[..., self.robot_model.get_joint_group_indices("waist")]
            left_leg_q = whole_q[..., self.robot_model.get_joint_group_indices("left_leg")]
            right_leg_q = whole_q[..., self.robot_model.get_joint_group_indices("right_leg")]
            left_hand_q = whole_q[..., self.robot_model.get_joint_group_indices("left_hand")]
            right_hand_q = whole_q[..., self.robot_model.get_joint_group_indices("right_hand")]
            
            # Add batch and time dimensions: (D,) or (B,D) or (B,T,D) -> (1, T, D)
            def add_batch_time_dims(arr, time_horizon):
                """Ensure array has shape (1, T, D) for policy. Handles (D,), (B, D), (B, T, D)."""
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    # (D,) -> (1, T, D)
                    arr = arr[np.newaxis, np.newaxis, ...]
                    arr = np.tile(arr, (1, time_horizon, 1))
                elif arr.ndim == 2:
                    # (B, D) -> (1, T, D): take first batch, tile in time
                    arr = arr[0:1, np.newaxis, :]  # (1, 1, D)
                    arr = np.tile(arr, (1, time_horizon, 1))
                elif arr.ndim == 3:
                    # (B, T, D) -> ensure (1, T, D)
                    arr = arr[0:1, :, :].astype(np.float32)
                    if arr.shape[1] != time_horizon:
                        arr = np.tile(arr[:, :1, :], (1, time_horizon, 1))
                return arr.astype(np.float32)
            
            obs["state.left_arm"] = add_batch_time_dims(left_arm_q, self.state_time_horizon)
            obs["state.right_arm"] = add_batch_time_dims(right_arm_q, self.state_time_horizon)
            obs["state.waist"] = add_batch_time_dims(waist_q, self.state_time_horizon)
            obs["state.left_leg"] = add_batch_time_dims(left_leg_q, self.state_time_horizon)
            obs["state.right_leg"] = add_batch_time_dims(right_leg_q, self.state_time_horizon)
            obs["state.left_hand"] = add_batch_time_dims(left_hand_q, self.state_time_horizon)
            obs["state.right_hand"] = add_batch_time_dims(right_hand_q, self.state_time_horizon)
        else:
            # If q shape doesn't match, skip state.* keys (policy may still work)
            if verbose:
                print(f"Warning: q shape {whole_q.shape[0]} != expected {self.robot_model.num_joints}, skipping state.* keys")

        # Add task description (must be tuple or list for policy server, never raw string)
        # Policy expects tuple[str] or list[str] with shape (B,) for batch
        if self.task_description is None:
            obs["annotation.human.task_description"] = ("",)
        elif isinstance(self.task_description, str):
            obs["annotation.human.task_description"] = (self.task_description,)
        elif isinstance(self.task_description, (tuple, list)):
            obs["annotation.human.task_description"] = tuple(self.task_description)
        else:
            obs["annotation.human.task_description"] = (str(self.task_description),)

        # Extract and map camera images
        self._add_camera_images(obs, og_obs)

        return obs

    def _compute_wrist_poses(self, q: np.ndarray) -> np.ndarray:
        """
        Compute left and right wrist poses (7D: pos + quat) from joint configuration.

        Args:
            q: Joint positions in Pinocchio joint order

        Returns:
            Concatenated wrist poses: [left_wrist_7d, right_wrist_7d] (14,)
        """
        # Use robot_model's forward kinematics to get end-effector poses
        try:
            # Cache forward kinematics first
            self.robot_model.cache_forward_kinematics(q, auto_clip=True)
            
            # Get frame placements (SE3 transforms)
            # Try common end-effector frame names
            left_frame_names = ["left_hand", "left_wrist", "left_end_effector"]
            right_frame_names = ["right_hand", "right_wrist", "right_end_effector"]
            
            left_placement = None
            right_placement = None
            
            for name in left_frame_names:
                try:
                    left_placement = self.robot_model.frame_placement(name)
                    break
                except (ValueError, KeyError):
                    continue
            
            for name in right_frame_names:
                try:
                    right_placement = self.robot_model.frame_placement(name)
                    break
                except (ValueError, KeyError):
                    continue
            
            if left_placement is not None and right_placement is not None:
                import pinocchio as pin
                # Extract position and quaternion from SE3
                left_pos = left_placement.translation
                left_quat = pin.Quaternion(left_placement.rotation).coeffs()  # [x,y,z,w]
                right_pos = right_placement.translation
                right_quat = pin.Quaternion(right_placement.rotation).coeffs()  # [x,y,z,w]
                
                return np.concatenate([left_pos, left_quat, right_pos, right_quat])
        except Exception as e:
            # Silently fail - wrist poses are optional, policy may still work
            pass

        # Fallback: return zeros
        return np.zeros(14)

    def _add_camera_images(self, obs: Dict[str, Any], og_obs: Dict[str, Any]) -> None:
        """
        Extract camera images from OmniGibson obs and add to policy obs format.

        Args:
            obs: Policy observation dict (modified in place)
            og_obs: OmniGibson observation dict
        """
        robot_obs = og_obs.get(self.robot_name, {})

        # Collect all camera RGB from robot and external sensors
        # OmniGibson may nest cameras: robot_obs["unitree_g1:d435_link:Camera:0"] = {"rgb": ...}
        camera_rgb = {}
        
        def _collect_rgb_recursive(d, prefix=""):
            """Recursively collect all 'rgb' values from nested dict."""
            if not isinstance(d, dict):
                return
            for k, v in d.items():
                if isinstance(v, dict):
                    if "rgb" in v:
                        full_key = f"{prefix}:{k}" if prefix else k
                        camera_rgb[full_key] = np.asarray(v["rgb"])
                    else:
                        # Recurse into nested dicts
                        new_prefix = f"{prefix}:{k}" if prefix else k
                        _collect_rgb_recursive(v, new_prefix)

        _collect_rgb_recursive(robot_obs, prefix="")
        
        # Also check external sensors
        external_obs = og_obs.get("external", {})
        for key, value in external_obs.items():
            if isinstance(value, dict) and "rgb" in value:
                camera_rgb[key] = np.asarray(value["rgb"])

        # Map OmniGibson camera keys to GR00T camera keys
        # For ego_view use _ego_camera_priority (head_camera first, then d435); fill each robocasa key only once.
        already_set_robocasa = set()
        sources = []
        for og_key in getattr(self, "_ego_camera_priority", ["head_camera", "unitree_g1:d435_link:Camera:0"]):
            sources.append((og_key, "robot0_oak_egoview"))
        sources.append(("unitree_g1:mid360_link:Camera:0", "robot0_rs_tppview"))

        for og_cam_key_pattern, robocasa_cam_key in sources:
            if robocasa_cam_key in already_set_robocasa:
                continue
            rgb = None
            if og_cam_key_pattern in camera_rgb:
                rgb = camera_rgb[og_cam_key_pattern]
            else:
                pattern_parts = og_cam_key_pattern.split(":")
                for cam_key in camera_rgb.keys():
                    if all(part in cam_key for part in pattern_parts if part):
                        rgb = camera_rgb[cam_key]
                        break

            if rgb is not None:
                # Ensure uint8 and correct shape
                if rgb.ndim == 3 and rgb.shape[-1] >= 3:
                    rgb_uint8 = rgb[:, :, :3].astype(np.uint8)
                    # Resize to match isaaclab_arena_g1 / GR00T evaluation (RS_VIEW_CAMERA_*)
                    target_h, target_w = self._camera_height, self._camera_width
                    if rgb_uint8.shape[:2] != (target_h, target_w):
                        rgb_uint8 = cv2.resize(rgb_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                    # Use CameraKeyMapper to get the mapped key (e.g., robot0_oak_egoview → ego_view)
                    if self.camera_key_mapper is not None:
                        mapped_key_result = self.camera_key_mapper.get_camera_config(robocasa_cam_key)
                        if mapped_key_result is not None:
                            mapped_key, _, _ = mapped_key_result
                            # Create _image key (single frame, H, W, C)
                            obs[f"{mapped_key}_image"] = rgb_uint8
                            # Create video.* key with batch and time dimensions (B=1, T, H, W, C)
                            # Policy expects shape (B, T, H, W, C), so add batch and time dims
                            # Repeat frame T times along time dimension
                            # rgb_uint8 is (H, W, C), add batch dim -> (1, H, W, C), then tile -> (1, T, H, W, C)
                            video_frames = np.tile(rgb_uint8[np.newaxis, np.newaxis, ...], (1, self.video_time_horizon, 1, 1, 1))  # (1, T, H, W, C)
                            assert video_frames.shape == (1, self.video_time_horizon, target_h, target_w, 3), \
                                f"Video shape mismatch: expected (1, {self.video_time_horizon}, {target_h}, {target_w}, 3), got {video_frames.shape}"
                            obs[f"video.{mapped_key}"] = video_frames
                        else:
                            # Fallback: use robocasa_cam_key directly if mapping not found
                            obs[f"{robocasa_cam_key}_image"] = rgb_uint8
                            video_frames = np.tile(rgb_uint8[np.newaxis, np.newaxis, ...], (1, self.video_time_horizon, 1, 1, 1))  # (1, T, H, W, C)
                            obs[f"video.{robocasa_cam_key}"] = video_frames
                    else:
                        # Fallback: use robocasa_cam_key directly if CameraKeyMapper not available
                        obs[f"{robocasa_cam_key}_image"] = rgb_uint8
                        video_frames = np.tile(rgb_uint8[np.newaxis, np.newaxis, ...], (1, self.video_time_horizon, 1, 1, 1))  # (1, T, H, W, C)
                        obs[f"video.{robocasa_cam_key}"] = video_frames
                    already_set_robocasa.add(robocasa_cam_key)
                else:
                    # Invalid shape, use zeros
                    if self.camera_key_mapper is not None:
                        mapped_key_result = self.camera_key_mapper.get_camera_config(robocasa_cam_key)
                        mapped_key = mapped_key_result[0] if mapped_key_result else robocasa_cam_key
                    else:
                        mapped_key = robocasa_cam_key
                    obs[f"{mapped_key}_image"] = np.zeros((self._camera_height, self._camera_width, 3), dtype=np.uint8)
                    obs[f"video.{mapped_key}"] = np.zeros((1, self.video_time_horizon, self._camera_height, self._camera_width, 3), dtype=np.uint8)  # (B=1, T, H, W, C)
            else:
                # Camera not found, use zeros
                if self.camera_key_mapper is not None:
                    mapped_key_result = self.camera_key_mapper.get_camera_config(robocasa_cam_key)
                    mapped_key = mapped_key_result[0] if mapped_key_result else robocasa_cam_key
                else:
                    mapped_key = robocasa_cam_key
                obs[f"{mapped_key}_image"] = np.zeros((self._camera_height, self._camera_width, 3), dtype=np.uint8)
                obs[f"video.{mapped_key}"] = np.zeros((1, self.video_time_horizon, self._camera_height, self._camera_width, 3), dtype=np.uint8)  # (B=1, T, H, W, C)


class GR00TToOmniGibsonActionAdapter:
    """Convert GR00T policy actions to OmniGibson action format."""

    def __init__(self, robot_model, robot_name: str = "unitree_g1"):
        """
        Args:
            robot_model: RobotModel instance
            robot_name: Name of the robot in OmniGibson action dict
        """
        self.robot_model = robot_model
        self.robot_name = robot_name
        self.og_action_dim = None  # OmniGibson action dimension (set externally)
        self.og_joint_names = None  # OmniGibson controllable joint names (set externally)
        self._joint_mapping = None  # Cached mapping from GR00T joints to OmniGibson joints
        
        # Import concat_action for converting action.left_arm, etc. to target_upper_body_pose
        # Inline it to avoid importing n1_utils (which has heavy dependencies)
        # We'll implement a simplified version

    def adapt(self, gr00t_action: Dict[str, Any], action_space=None) -> Dict[str, Any]:
        """
        Convert GR00T policy action to OmniGibson action format.

        Args:
            gr00t_action: Policy action dict with keys like:
                - "q" (full joint positions in Pinocchio joint order)
                - "target_upper_body_pose" (upper body only)
                - "action.left_arm", "action.right_arm", etc. (from Gr00tSimPolicyWrapper)
            action_space: Optional gym.Space to infer action format (if None, uses Dict format)

        Returns:
            OmniGibson action dict. Format depends on action_space:
            - If Dict: {robot_name: np.ndarray} where the array is joint positions in
              **og_joint_names order** (same as robot.joint_names when set by run script).
              Length is og_action_dim (may be less than robot_model.num_joints if OG has fewer controllable joints).
            - If Box: same array as above (flat joint positions).
        """
        # Handle different action formats from policy wrapper
        if "q" in gr00t_action:
            # Direct full joint positions
            q = np.asarray(gr00t_action["q"])
        elif "target_upper_body_pose" in gr00t_action:
            # Upper body only - need current_q to reconstruct full q
            raise ValueError(
                "Action adapter needs current joint state to reconstruct full q from target_upper_body_pose. "
                "Use adapt_with_current_state() instead."
            )
        elif any(key.startswith("action.") for key in gr00t_action.keys()):
            # Policy wrapper format: action.left_arm, action.right_arm, etc.
            # Convert to target_upper_body_pose first (simplified concat_action)
            processed_goal = {}
            for key, value in gr00t_action.items():
                if key.startswith("action."):
                    processed_goal[key.replace("action.", "")] = np.asarray(value)
            
            # Extract first action value to get shape (B, T, D) or (T, D) or (D,)
            if not processed_goal:
                raise ValueError(f"No action.* keys found in action dict: {list(gr00t_action.keys())}")
            
            first_value = next(iter(processed_goal.values()))
            # Handle different shapes: (B, T, D), (T, D), or (D,)
            if first_value.ndim == 3:  # (B, T, D)
                batch_shape = first_value.shape[:2]
            elif first_value.ndim == 2:  # (T, D)
                batch_shape = (1, first_value.shape[0])
            else:  # (D,)
                batch_shape = (1, 1)
            
            # Create full action array
            action = np.zeros(batch_shape + (self.robot_model.num_joints,))
            
            # Fill in joint groups
            for joint_group, value in processed_goal.items():
                value = np.asarray(value)
                # Handle shape: ensure it has batch/time dims
                if value.ndim == 1:
                    value = value[np.newaxis, np.newaxis, ...]  # (1, 1, D)
                elif value.ndim == 2:
                    value = value[np.newaxis, ...]  # (1, T, D)
                # value is now (B, T, D) or (1, T, D)
                
                try:
                    indices = self.robot_model.get_joint_group_indices(joint_group)
                    action[..., indices] = value
                except (ValueError, KeyError):
                    # Skip unknown joint groups (like navigate_command, base_height_command)
                    continue
            
            # Extract upper body indices
            upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
            target_upper_body = action[..., upper_body_indices]
            
            # For now, raise error - need current_q to reconstruct full q
            raise ValueError(
                "Action adapter needs current joint state to reconstruct full q from action.* keys. "
                "Use adapt_with_current_state() instead."
            )
        else:
            raise ValueError(f"GR00T action must contain 'q', 'target_upper_body_pose', or 'action.*' keys, got keys: {gr00t_action.keys()}")

        # Ensure q is the right length
        if len(q) != self.robot_model.num_joints:
            raise ValueError(
                f"Action q length {len(q)} does not match robot_model.num_joints {self.robot_model.num_joints}"
            )

        # Map to OmniGibson action space (may have fewer joints)
        q_mapped = self._map_joints_to_og_action_space(q, action_space)

        # OmniGibson action format:
        # - If action_space is Dict: return {robot_name: 1D_array}
        # - If action_space is Box: return 1D_array directly
        import gymnasium as gym
        if action_space is not None and isinstance(action_space, gym.spaces.Dict):
            return {self.robot_name: q_mapped.astype(np.float32)}
        else:
            return q_mapped.astype(np.float32)

    def adapt_with_current_state(
        self, gr00t_action: Dict[str, Any], current_q: np.ndarray, action_space=None
    ) -> Dict[str, Any]:
        """
        Convert GR00T action using current joint state (for target_upper_body_pose or action.* keys).

        Args:
            gr00t_action: Policy action dict
            current_q: Current joint positions (to fill in non-upper-body joints)
            action_space: Optional gym.Space to infer action format

        Returns:
            OmniGibson action dict or array
        """
        # Handle different action formats
        if "q" in gr00t_action:
            # Direct full joint positions
            q = np.asarray(gr00t_action["q"])
            # Handle batch/time dimensions: take first batch, first time step
            if q.ndim == 3:  # (B, T, D)
                q = q[0, 0, :]  # Take first batch, first time step
            elif q.ndim == 2:  # (T, D)
                q = q[0, :]  # Take first time step
        elif "target_upper_body_pose" in gr00t_action:
            # Reconstruct full q: upper_body from action, rest from current_q
            upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
            target_upper_body = np.asarray(gr00t_action["target_upper_body_pose"])
            # Handle batch/time dimensions
            if target_upper_body.ndim == 3:  # (B, T, D)
                target_upper_body = target_upper_body[0, 0, :]
            elif target_upper_body.ndim == 2:  # (T, D)
                target_upper_body = target_upper_body[0, :]
            
            q = current_q.copy()
            if len(target_upper_body) == len(upper_body_indices):
                q[upper_body_indices] = target_upper_body
            else:
                # If shape doesn't match, try to pad/truncate
                if len(target_upper_body) < len(upper_body_indices):
                    target_upper_body = np.pad(target_upper_body, (0, len(upper_body_indices) - len(target_upper_body)))
                else:
                    target_upper_body = target_upper_body[:len(upper_body_indices)]
                q[upper_body_indices] = target_upper_body
        elif any(key.startswith("action.") for key in gr00t_action.keys()):
            # Policy wrapper format: action.left_arm, action.right_arm, etc.
            # Convert to target_upper_body_pose, then reconstruct full q
            processed_goal = {}
            for key, value in gr00t_action.items():
                if key.startswith("action."):
                    processed_goal[key.replace("action.", "")] = np.asarray(value)
            
            if not processed_goal:
                raise ValueError(f"No action.* keys found in action dict: {list(gr00t_action.keys())}")
            
            # Extract first action value to determine shape
            first_value = next(iter(processed_goal.values()))
            # Handle different shapes: take first batch, first time step if needed
            if first_value.ndim == 3:  # (B, T, D)
                take_slice = (0, 0, slice(None))
            elif first_value.ndim == 2:  # (T, D)
                take_slice = (0, slice(None))
            else:  # (D,)
                take_slice = (slice(None),)
            
            # Create full action array (single timestep)
            action = np.zeros(self.robot_model.num_joints)
            
            # Fill in joint groups (taking first batch/time step)
            for joint_group, value in processed_goal.items():
                value = np.asarray(value)
                # Take first batch/time step
                if value.ndim == 3:
                    value = value[0, 0, :]
                elif value.ndim == 2:
                    value = value[0, :]
                
                try:
                    indices = self.robot_model.get_joint_group_indices(joint_group)
                    if len(value) == len(indices):
                        action[indices] = value
                except (ValueError, KeyError):
                    # Skip unknown joint groups (like navigate_command, base_height_command)
                    continue
            
            # Extract upper body and reconstruct full q
            upper_body_indices = self.robot_model.get_joint_group_indices("upper_body")
            target_upper_body = action[upper_body_indices]
            
            q = current_q.copy()
            q[upper_body_indices] = target_upper_body
        else:
            raise ValueError(f"GR00T action must contain 'q', 'target_upper_body_pose', or 'action.*' keys, got keys: {gr00t_action.keys()}")

        # Ensure q length matches
        if len(q) != self.robot_model.num_joints:
            if len(q) < self.robot_model.num_joints:
                q = np.pad(q, (0, self.robot_model.num_joints - len(q)))
            else:
                q = q[:self.robot_model.num_joints]

        # Map to OmniGibson action space (may have fewer joints)
        q_mapped = self._map_joints_to_og_action_space(q, action_space)

        # OmniGibson action format:
        # - If action_space is Dict: return {robot_name: 1D_array}
        # - If action_space is Box: return 1D_array directly
        import gymnasium as gym
        if action_space is not None and isinstance(action_space, gym.spaces.Dict):
            return {self.robot_name: q_mapped.astype(np.float32)}
        else:
            return q_mapped.astype(np.float32)
    
    def _map_joints_to_og_action_space(self, q: np.ndarray, action_space=None) -> np.ndarray:
        """
        Map full GR00T joint positions to OmniGibson's action space.
        OmniGibson may have fewer controllable joints (e.g., excludes finger joints).
        
        Args:
            q: Full joint positions (self.robot_model.num_joints)
            action_space: Optional gym.Space to infer action format
            
        Returns:
            Mapped joint positions (og_action_dim or len(q) if no mapping needed)
        """
        # If no action dimension specified, return full q
        if self.og_action_dim is None:
            return q
        
        # If dimensions match, no mapping needed
        if len(q) == self.og_action_dim:
            return q
        
        # Try to create mapping based on joint names
        if self._joint_mapping is None and self.og_joint_names is not None:
            # Get GR00T joint names from robot model (ordered by DOF index)
            try:
                # RobotModel stores joint names in joint_to_dof_index dict
                if hasattr(self.robot_model, 'joint_to_dof_index'):
                    # Sort joint names by their DOF index to match q order
                    joint_to_dof = self.robot_model.joint_to_dof_index
                    gr00t_joint_names = sorted(joint_to_dof.keys(), key=lambda name: joint_to_dof[name])
                elif hasattr(self.robot_model, 'pinocchio_wrapper'):
                    # Fallback: get from pinocchio model (skip universe and floating base)
                    model = self.robot_model.pinocchio_wrapper.model
                    start_idx = 2 if self.robot_model.is_floating_base_model else 1
                    gr00t_joint_names = model.names[start_idx:]
                else:
                    gr00t_joint_names = None
            except Exception as e:
                gr00t_joint_names = None
            
            if gr00t_joint_names and len(gr00t_joint_names) == len(q):
                # Create mapping: find indices of OG joints in GR00T joint list
                mapping_indices = []
                for og_joint_name in self.og_joint_names:
                    # Try exact match first
                    if og_joint_name in gr00t_joint_names:
                        mapping_indices.append(gr00t_joint_names.index(og_joint_name))
                    else:
                        # Try partial match (OG might use different naming)
                        # Common patterns: OG might use "left_hand_joint_1" vs GR00T "left_hand_1"
                        matched = False
                        for i, gr00t_name in enumerate(gr00t_joint_names):
                            # Normalize names for comparison
                            og_norm = og_joint_name.lower().replace("_", "").replace("-", "")
                            gr00t_norm = gr00t_name.lower().replace("_", "").replace("-", "")
                            if og_norm in gr00t_norm or gr00t_norm in og_norm:
                                mapping_indices.append(i)
                                matched = True
                                break
                        if not matched:
                            # If can't match, skip this joint (will use default/current value)
                            pass
                
                if len(mapping_indices) == self.og_action_dim:
                    self._joint_mapping = np.array(mapping_indices)
                elif len(mapping_indices) > 0:
                    # Partial mapping - pad with zeros or repeat last valid index
                    while len(mapping_indices) < self.og_action_dim:
                        mapping_indices.append(mapping_indices[-1] if mapping_indices else 0)
                    self._joint_mapping = np.array(mapping_indices[:self.og_action_dim])
        
        # Apply mapping if available
        if self._joint_mapping is not None and len(self._joint_mapping) == self.og_action_dim:
            return q[self._joint_mapping]
        
        # Fallback: Use RobotModel's get_body_actuated_joint_indices() to exclude hand joints
        # This is the correct way to get body-only joints regardless of joint order
        if len(q) > self.og_action_dim:
            try:
                # Get body joint indices (excludes hand joints)
                body_indices = self.robot_model.get_body_actuated_joint_indices()
                if len(body_indices) == self.og_action_dim:
                    return q[body_indices]
                elif len(body_indices) > 0:
                    # If body_indices count matches expected, use them
                    # Otherwise take first N body joints
                    return q[body_indices[:self.og_action_dim]]
            except Exception:
                # If method fails, fall back to simple slicing (may be incorrect)
                pass
            
            # Last resort: take first N joints (may be incorrect if hand joints are interspersed)
            return q[:self.og_action_dim]
        
        # If OG expects more joints (unlikely), pad with zeros
        if len(q) < self.og_action_dim:
            return np.pad(q, (0, self.og_action_dim - len(q)), mode='constant')
        
        return q
