"""
Robot interface module for the teleoperation system.
Supports both SO100 (Feetech, 6-DOF) and reBot B601-DM (Damiao CAN, 7-DOF).
"""

import numpy as np
import time
import logging
import os
import sys
import contextlib
from collections import deque
from typing import Optional, Dict, Tuple


def _rot_angle(rot_a: np.ndarray, rot_b: np.ndarray) -> float:
    """Geodesic angle (radians) between two rotation matrices."""
    r = np.asarray(rot_a).T @ np.asarray(rot_b)
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))

from ..driver.so100 import SOFollower, SOFollowerConfig as SOFollowerRobotConfig

from ..config import TelegripConfig, GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE, _config_data
from .kinematics import ForwardKinematics, IKSolver

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output at the file descriptor level."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Save original file descriptors
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    
    try:
        # Open devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        
        # Redirect stdout and stderr to devnull
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        
        yield
        
    finally:
        # Restore original file descriptors
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        
        # Close saved file descriptors
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


class RobotInterface:
    """High-level interface for robot control — supports SO100 (6-DOF) and reBot (7-DOF)."""

    def __init__(self, config: TelegripConfig):
        self.config = config
        self.left_robot = None
        self.right_robot = None
        self.is_connected = False
        self.is_engaged = False

        # Individual arm connection status
        self.left_arm_connected = False
        self.right_arm_connected = False

        # Joint count for the selected robot type
        self._num_joints = config.num_joints

        # Joint state
        self.left_arm_angles = np.zeros(self._num_joints)
        self.right_arm_angles = np.zeros(self._num_joints)

        # Joint limits (will be set by visualizer)
        self.joint_limits_min_deg = np.full(self._num_joints, -180.0)
        self.joint_limits_max_deg = np.full(self._num_joints, 180.0)

        # Kinematics solvers (will be set after PyBullet setup)
        self.fk_solvers = {'left': None, 'right': None}
        self.ik_solvers = {'left': None, 'right': None}

        # Control timing
        self.last_send_time = 0

        # Error tracking - separate for each arm
        self.left_arm_errors = 0
        self.right_arm_errors = 0
        self.general_errors = 0
        self.max_arm_errors = 3
        self.max_general_errors = 8

        # Safe home positions for shutdown
        home = np.array(config.home_position, dtype=float)
        self.initial_left_arm = home.copy()
        self.initial_right_arm = home.copy()
    
    def setup_robot_configs(self):
        """Create robot configurations for enabled arms (SO100 or reBot)."""
        logger.info(
            f"Setting up {self.config.robot_type.upper()} robot configs "
            f"with ports: {self.config.follower_ports}"
        )
        if self.config.is_rebot:
            return self._setup_rebot_configs()
        return self._setup_so100_configs()

    def _setup_so100_configs(self):
        calib = getattr(self.config, 'calibration_paths', {}) or {}
        left_config = None
        if self.config.arm_enabled.get("left", True):
            left_config = SOFollowerRobotConfig(
                port=self.config.follower_ports["left"],
                id="left_follower",
                use_degrees=True,
                disable_torque_on_disconnect=True,
                calibration_path=calib.get("left"),
            )
        right_config = None
        if self.config.arm_enabled.get("right", True):
            right_config = SOFollowerRobotConfig(
                port=self.config.follower_ports["right"],
                id="right_follower",
                use_degrees=True,
                disable_torque_on_disconnect=True,
                calibration_path=calib.get("right"),
            )
        return left_config, right_config

    def _setup_rebot_configs(self):
        from ..driver.rebot import RebotFollower, RebotFollowerConfig
        raw = _config_data["robot"]

        def _arm_cfg(arm_key: str, arm_id: str):
            arm = raw.get(arm_key, {})
            vel = float(self.config.rebot_motor_velocity_deg_s)
            return RebotFollowerConfig(
                port=self.config.follower_ports[
                    "left" if arm_key == "left_arm" else "right"
                ],
                id=arm_id,
                can_adapter=arm.get("can_adapter", "damiao"),
                dm_serial_baud=arm.get("dm_serial_baud", 921_600),
                vel_deg_s=[vel] * 7,
                control_mode=self.config.rebot_control_mode,
                mit_kp=list(self.config.rebot_mit_kp),
                mit_kd=list(self.config.rebot_mit_kd),
            )

        left_config = None
        if self.config.arm_enabled.get("left", True):
            left_config = _arm_cfg("left_arm", "left_rebot")
        right_config = None
        if self.config.arm_enabled.get("right", True):
            right_config = _arm_cfg("right_arm", "right_rebot")
        return left_config, right_config
    
    def connect(self) -> bool:
        """Connect to robot hardware."""
        if self.is_connected:
            logger.info("Robot interface already connected")
            return True
        
        if not self.config.enable_robot:
            logger.info("Robot interface disabled in config")
            self.is_connected = True  # Mark as "connected" for testing
            return True
        
        # Setup suppression if requested
        should_suppress = (self.config.log_level == "warning" or 
                          self.config.log_level == "critical" or 
                          self.config.log_level == "error")
        
        try:
            left_config, right_config = self.setup_robot_configs()
            if not should_suppress:
                logger.info("Connecting to robot...")
            
            # Select driver class
            if self.config.is_rebot:
                from ..driver.rebot import RebotFollower as DriverClass
            else:
                DriverClass = SOFollower

            # Connect left arm
            if left_config is not None:
                try:
                    if should_suppress:
                        with suppress_stdout_stderr():
                            self.left_robot = DriverClass(left_config)
                            self.left_robot.connect()
                    else:
                        self.left_robot = DriverClass(left_config)
                        self.left_robot.connect()
                    self.left_arm_connected = True
                    logger.info("✅ Left arm connected successfully")
                except Exception as e:
                    logger.error(f"❌ Left arm connection failed: {e}")
                    self.left_arm_connected = False
            else:
                logger.info("Left arm disabled in configuration")
                self.left_arm_connected = False

            # Connect right arm
            if right_config is not None:
                try:
                    if should_suppress:
                        with suppress_stdout_stderr():
                            self.right_robot = DriverClass(right_config)
                            self.right_robot.connect()
                    else:
                        self.right_robot = DriverClass(right_config)
                        self.right_robot.connect()
                    self.right_arm_connected = True
                    logger.info("✅ Right arm connected successfully")
                except Exception as e:
                    logger.error(f"❌ Right arm connection failed: {e}")
                    self.right_arm_connected = False
            else:
                logger.info("Right arm disabled in configuration")
                self.right_arm_connected = False
                
            # Mark as connected if at least one arm is connected
            self.is_connected = self.left_arm_connected or self.right_arm_connected
            
            if self.is_connected:
                # Initialize joint states
                self._read_initial_state()
                logger.info(f"🤖 Robot interface connected: Left={self.left_arm_connected}, Right={self.right_arm_connected}")
            else:
                logger.error("❌ Failed to connect any robot arms")
                
            return self.is_connected
            
        except Exception as e:
            logger.error(f"❌ Robot connection failed with exception: {e}")
            self.is_connected = False
            return False
    
    def _read_initial_state(self):
        """Read initial joint state from robot."""
        joint_names = self.config.joint_names
        try:
            if self.left_robot and self.left_arm_connected:
                observation = self.left_robot.get_observation()
                if observation:
                    self.left_arm_angles = np.array(
                        [observation[f"{j}.pos"] for j in joint_names]
                    )
                    logger.info(f"Left arm initial state: {self.left_arm_angles.round(1)}")

            if self.right_robot and self.right_arm_connected:
                observation = self.right_robot.get_observation()
                if observation:
                    self.right_arm_angles = np.array(
                        [observation[f"{j}.pos"] for j in joint_names]
                    )
                    logger.info(f"Right arm initial state: {self.right_arm_angles.round(1)}")

        except Exception as e:
            logger.error(f"Error reading initial state: {e}")
    
    def setup_kinematics(self, physics_client, robot_ids: Dict, joint_indices: Dict,
                        end_effector_link_indices: Dict, joint_limits_min_deg: np.ndarray,
                        joint_limits_max_deg: np.ndarray):
        """Setup kinematics solvers using PyBullet components for both arms."""
        self.joint_limits_min_deg = joint_limits_min_deg.copy()
        self.joint_limits_max_deg = joint_limits_max_deg.copy()
        
        # Setup solvers for both arms
        for arm in ['left', 'right']:
            self.fk_solvers[arm] = ForwardKinematics(
                physics_client, robot_ids[arm], joint_indices[arm], end_effector_link_indices[arm]
            )
            
            self.ik_solvers[arm] = IKSolver(
                physics_client, robot_ids[arm], joint_indices[arm], end_effector_link_indices[arm],
                joint_limits_min_deg, joint_limits_max_deg, arm_name=arm
            )
        
        logger.info("Kinematics solvers initialized for both arms")
    
    def get_current_end_effector_position(self, arm: str) -> np.ndarray:
        """Get current end effector position for specified arm."""
        if arm == "left":
            angles = self.left_arm_angles
        elif arm == "right":
            angles = self.right_arm_angles
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        if self.fk_solvers[arm]:
            position, _ = self.fk_solvers[arm].compute(angles)
            return position
        else:
            default_position = np.array([0.2, 0.0, 0.15])
            return default_position
    
    def solve_ik(self, arm: str, target_position: np.ndarray, 
                 target_orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """Solve inverse kinematics for specified arm."""
        if arm == "left":
            current_angles = self.left_arm_angles
        elif arm == "right":
            current_angles = self.right_arm_angles
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        if self.ik_solvers[arm]:
            return self.ik_solvers[arm].solve(target_position, target_orientation, current_angles)
        else:
            return current_angles[:3]  # Return current angles if no IK solver
    
    def clamp_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
        """Clamp joint angles to safe limits with margins for problem joints."""
        # Create a copy to avoid modifying the original
        processed_angles = joint_angles.copy()
        
        # First, normalize angles that can wrap around (like shoulder_pan)
        # Check if first joint (shoulder_pan) is outside limits but can be wrapped
        shoulder_pan_idx = 0
        shoulder_pan_angle = processed_angles[shoulder_pan_idx]
        min_limit = self.joint_limits_min_deg[shoulder_pan_idx]  # -120.3°
        max_limit = self.joint_limits_max_deg[shoulder_pan_idx]  # +120.3°
        
        # Try to wrap the angle to an equivalent angle within limits
        if shoulder_pan_angle < min_limit or shoulder_pan_angle > max_limit:
            # Try wrapping by ±360°
            for offset in [-360.0, 360.0]:
                wrapped_angle = shoulder_pan_angle + offset
                if min_limit <= wrapped_angle <= max_limit:
                    logger.debug(f"Wrapped shoulder_pan from {shoulder_pan_angle:.1f}° to {wrapped_angle:.1f}°")
                    processed_angles[shoulder_pan_idx] = wrapped_angle
                    break
        
        # Apply standard joint limits to all joints
        return np.clip(processed_angles, self.joint_limits_min_deg, self.joint_limits_max_deg)
    
    def update_arm_angles(self, arm: str, ik_angles: np.ndarray,
                         wrist_flex: float, wrist_roll: float, gripper: float,
                         wrist_yaw: Optional[float] = None):
        """Update joint angles with IK solution and direct wrist/gripper control."""
        if arm == "left":
            target_angles = self.left_arm_angles
        elif arm == "right":
            target_angles = self.right_arm_angles
        else:
            raise ValueError(f"Invalid arm: {arm}")

        wfi = self.config.wrist_flex_index
        wyi = self.config.wrist_yaw_index
        wri = self.config.wrist_roll_index
        gi  = self.config.gripper_index

        # First 3 joints set by IK
        target_angles[:3] = ik_angles

        # Wrist joints — directly commanded
        target_angles[wfi] = wrist_flex
        if wyi is not None and wrist_yaw is not None:
            target_angles[wyi] = wrist_yaw
        target_angles[wri] = wrist_roll

        # Gripper — sort limits so clip works regardless of whether closed > open
        # (reBot: open=0°, closed=-45°  →  clip to [-45, 0])
        # (SO100: open=0°, closed=+45°  →  clip to [0, 45])
        _g_lo = min(GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE)
        _g_hi = max(GRIPPER_OPEN_ANGLE, GRIPPER_CLOSED_ANGLE)
        target_angles[gi] = np.clip(gripper, _g_lo, _g_hi)

        clamped_angles = self.clamp_joint_angles(target_angles)
        clamped_angles[gi] = target_angles[gi]  # preserve intentional gripper value

        if arm == "left":
            self.left_arm_angles = clamped_angles
        else:
            self.right_arm_angles = clamped_angles
    
    # ------------------------------------------------------------------ #
    # reBot: official 6-DOF pose FK/IK (Pinocchio). SO100 does not use these.
    # ------------------------------------------------------------------ #

    def _ensure_rebot_kin(self):
        """Lazily build the official-IK adapter (reBot only)."""
        if not self.config.is_rebot:
            return None
        if getattr(self, "_rebot_kin", None) is None:
            from .rebot_kinematics import RebotKinematics
            self._rebot_kin = RebotKinematics()
        return self._rebot_kin

    def get_current_end_effector_pose(self, arm: str) -> Tuple[np.ndarray, np.ndarray]:
        """reBot: end-effector (position (3,), rotation (3x3)) via official FK.

        Uses the current commanded joint angles (first 6 joints, degrees).
        """
        kin = self._ensure_rebot_kin()
        angles = self.left_arm_angles if arm == "left" else self.right_arm_angles
        q6 = np.deg2rad(np.asarray(angles[:6], dtype=float))
        return kin.fk(q6)

    def _solve_ik_deg(self, arm: str, target_position: np.ndarray,
                      target_rotation: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Official 6-DOF IK → joint angles (degrees), warm-started from the
        current commanded joints. No state change."""
        kin = self._ensure_rebot_kin()
        angles = self.left_arm_angles if arm == "left" else self.right_arm_angles
        seed = np.deg2rad(np.asarray(angles[:6], dtype=float))
        q6, success, err = kin.solve(target_position, target_rotation, seed)
        if not success:
            logger.debug(f"reBot IK did not converge (err={err:.4f}) for {arm} arm")
        return np.rad2deg(q6), success

    def _set_arm_joints6(self, arm: str, q6_deg: np.ndarray) -> None:
        """Write joints 1-6 (degrees), clamp to limits, preserve the gripper."""
        angles = self.left_arm_angles if arm == "left" else self.right_arm_angles
        new_angles = angles.copy()
        new_angles[:6] = np.asarray(q6_deg, dtype=float)
        gi = self.config.gripper_index
        clamped = self.clamp_joint_angles(new_angles)
        clamped[gi] = angles[gi]  # preserve gripper
        if arm == "left":
            self.left_arm_angles = clamped
        else:
            self.right_arm_angles = clamped

    def update_arm_from_pose(self, arm: str, target_position: np.ndarray,
                             target_rotation: np.ndarray) -> bool:
        """reBot: solve the official 6-DOF IK for a full pose and store the angles.

        The gripper (joint 7) is preserved; only joints 1-6 are set from IK.
        (Direct IK→set path used by tests; the live teleop path uses the OTG.)
        """
        angles = (self.left_arm_angles if arm == "left" else self.right_arm_angles)
        q6_deg, success = self._solve_ik_deg(arm, target_position, target_rotation)

        new_angles = angles.copy()
        new_angles[:6] = q6_deg

        # Per-joint speed cap (sensitivity tuning): limit how far each joint may
        # move this cycle so over-sensitive joints (e.g. pan, wrist) track the
        # IK target gently instead of twitching. Converges over a few cycles.
        mv = getattr(self.config, "rebot_joint_max_vel_deg_s", None)
        if mv:
            dt = float(self.config.send_interval)
            prev6 = np.asarray(angles[:6], dtype=float)
            max_step = np.array(
                [float(mv[j]) * dt if j < len(mv) else 1e9 for j in range(6)]
            )
            step = np.clip(new_angles[:6] - prev6, -max_step, max_step)
            new_angles[:6] = prev6 + step

        gi = self.config.gripper_index
        clamped = self.clamp_joint_angles(new_angles)
        clamped[gi] = angles[gi]  # preserve gripper

        if arm == "left":
            self.left_arm_angles = clamped
        else:
            self.right_arm_angles = clamped
        return success

    def _rebot_tracking_state(self):
        if not hasattr(self, "_rebot_cmd_pose"):
            self._rebot_cmd_pose = {"left": None, "right": None}
            self._rebot_path = {"left": deque(), "right": deque()}
        return self._rebot_cmd_pose, self._rebot_path

    # ------------------------------------------------------------------ #
    # Jerk-limited online trajectory generation (Ruckig) on joint commands #
    # ------------------------------------------------------------------ #

    def _ensure_otg(self):
        """Lazily build a per-arm Ruckig 6-DOF online trajectory generator."""
        if getattr(self, "_otg", None) is None:
            from ruckig import Ruckig, InputParameter, OutputParameter
            dt = float(self.config.send_interval)
            mv = getattr(self.config, "rebot_joint_max_vel_deg_s", None) or [250.0] * 6
            vmax = [float(mv[j]) if j < len(mv) else 250.0 for j in range(6)]
            amax = [float(self.config.rebot_otg_max_accel_deg_s2)] * 6
            jmax = [float(self.config.rebot_otg_max_jerk_deg_s3)] * 6
            self._otg = {}
            self._otg_inp = {}
            self._otg_out = {}
            self._otg_state = {"left": None, "right": None}
            for a in ("left", "right"):
                self._otg[a] = Ruckig(6, dt)
                inp = InputParameter(6)
                inp.max_velocity = vmax
                inp.max_acceleration = amax
                inp.max_jerk = jmax
                self._otg_inp[a] = inp
                self._otg_out[a] = OutputParameter(6)
        return self._otg

    def _otg_reset(self, arm: str) -> None:
        """Re-seed the OTG state at the current joints with zero vel/accel."""
        self._ensure_otg()
        cur = np.asarray(
            (self.left_arm_angles if arm == "left" else self.right_arm_angles)[:6],
            dtype=float,
        )
        self._otg_state[arm] = (list(cur), [0.0] * 6, [0.0] * 6)

    def _otg_step(self, arm: str, q_target_deg: np.ndarray) -> np.ndarray:
        """Advance one jerk-limited step toward q_target (degrees); returns the
        new commanded joint angles (degrees)."""
        self._ensure_otg()
        if self._otg_state[arm] is None:
            self._otg_reset(arm)
        inp = self._otg_inp[arm]
        out = self._otg_out[arm]
        pos, vel, acc = self._otg_state[arm]
        inp.current_position = pos
        inp.current_velocity = vel
        inp.current_acceleration = acc
        inp.target_position = [float(x) for x in q_target_deg]
        inp.target_velocity = [0.0] * 6
        inp.target_acceleration = [0.0] * 6
        try:
            self._otg[arm].update(inp, out)
            self._otg_state[arm] = (
                list(out.new_position), list(out.new_velocity), list(out.new_acceleration)
            )
        except Exception as e:
            logger.debug(f"OTG step failed ({e}); passing IK target through")
            self._otg_state[arm] = ([float(x) for x in q_target_deg], [0.0] * 6, [0.0] * 6)
        return np.asarray(self._otg_state[arm][0], dtype=float)

    def anchor_rebot_tracking(self, arm: str) -> None:
        """Reset the path follower to the current arm pose (grip start).

        Clears the waypoint queue, re-seeds the commanded pose and the OTG state
        to where the arm actually is, so tracking begins with a zero step.
        """
        cmd_pose, path = self._rebot_tracking_state()
        pos, rot = self.get_current_end_effector_pose(arm)
        cmd_pose[arm] = (pos.copy(), rot.copy())
        path[arm].clear()
        self._otg_reset(arm)

    def freeze_rebot(self, arm: str) -> None:
        """Watchdog freeze: drop any buffered path so the arm stops advancing and
        holds at its current commanded pose (the OTG decelerates to a stop)."""
        if not self.config.is_rebot:
            return
        _, path = self._rebot_tracking_state()
        path[arm].clear()

    def add_waypoint(self, arm: str, pos: np.ndarray, rot: np.ndarray,
                     min_lin: float, min_ang: float, max_len: int) -> None:
        """Append a hand-pose waypoint to the arm's path, sampled by minimum
        distance so the queue captures the path shape without redundant points.

        If the hand has not moved at least ``min_lin`` m / ``min_ang`` rad from
        the last waypoint, the last waypoint is updated in place (keeps the
        target current without growing the queue). The queue length is capped to
        bound how far the arm can lag behind a fast hand motion.
        """
        _, path = self._rebot_tracking_state()
        q = path[arm]
        pos = np.asarray(pos, dtype=float).copy()
        rot = np.asarray(rot, dtype=float).copy()
        if q:
            last_pos, last_rot = q[-1]
            if (np.linalg.norm(pos - last_pos) < min_lin and
                    _rot_angle(last_rot, rot) < min_ang):
                q[-1] = (pos, rot)
                return
        q.append((pos, rot))
        while len(q) > max_len:
            q.popleft()  # drop oldest unreached point → bound lag

    def track_path(self, arm: str, max_lin: float, max_ang: float) -> bool:
        """Advance the commanded pose along the queued waypoint polyline by a
        bounded per-cycle step, popping waypoints as they are reached, then solve
        IK on the resulting intermediate pose.

        The arm thus traces the actual hand path (not just the latest target),
        with each joint following a smooth continuous trajectory.
        """
        kin = self._ensure_rebot_kin()
        cmd_pose, path = self._rebot_tracking_state()
        if cmd_pose[arm] is None:
            cmd_pose[arm] = self.get_current_end_effector_pose(arm)
        cmd_pos, cmd_rot = cmd_pose[arm]
        q = path[arm]

        budget = max_lin
        # Walk the polyline: consume reached waypoints, step into the next.
        while q:
            front_pos, front_rot = q[0]
            d = float(np.linalg.norm(front_pos - cmd_pos))
            if len(q) > 1 and d <= budget:
                # Reached this waypoint; snap to it and continue with remaining budget.
                cmd_pos, cmd_rot = front_pos, front_rot
                budget -= d
                q.popleft()
                continue
            # Step toward the (front / final) waypoint by the remaining budget.
            cmd_pos, cmd_rot = kin.step_pose(
                cmd_pos, cmd_rot, front_pos, front_rot, budget, max_ang
            )
            break

        cmd_pose[arm] = (cmd_pos, cmd_rot)

        # IK to the intermediate (path-followed) pose, then jerk-limited OTG so
        # the joint command is acceleration/jerk-bounded → smooth at the source.
        q_target, success = self._solve_ik_deg(arm, cmd_pos, cmd_rot)
        q_cmd = self._otg_step(arm, q_target)
        self._set_arm_joints6(arm, q_cmd)
        return success

    def sync_commanded_to_actual(self, arm: str) -> None:
        """Re-anchor commanded joint angles to the actual hardware reading.

        Called at grip start (reBot) so the FK origin and the IK seed match the
        real arm rather than the last open-loop commanded values.
        """
        actual = self.get_actual_arm_angles(arm)
        if actual is None:
            return
        actual = np.asarray(actual, dtype=float)
        if arm == "left":
            self.left_arm_angles = actual.copy()
        else:
            self.right_arm_angles = actual.copy()

    def engage(self) -> bool:
        """Engage robot motors (start sending commands)."""
        if not self.is_connected:
            logger.warning("Cannot engage robot: not connected")
            return False
        
        self.is_engaged = True
        logger.info("🔌 Robot motors ENGAGED - commands will be sent")
        return True
    
    def disengage(self) -> bool:
        """Disengage robot motors (stop sending commands)."""
        if not self.is_connected:
            logger.info("Robot already disconnected")
            return True
        
        try:
            # Return to safe position before disengaging
            self.return_to_initial_position()
            
            # Disable torque
            self.disable_torque()
            
            self.is_engaged = False
            logger.info("🔌 Robot motors DISENGAGED - commands stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error disengaging robot: {e}")
            return False
    
    def _send_to_driver(self, robot, arm: str) -> None:
        """Push the current commanded joints to one driver, in the active mode.

        pos_vel (SO100 + default reBot): position dict.
        mit (reBot impedance): position + velocity (from OTG) + gravity torque.
        """
        joint_names = self.config.joint_names
        angles = self.left_arm_angles if arm == "left" else self.right_arm_angles

        if self.config.is_rebot and self.config.rebot_control_mode == "mit":
            positions = {j: float(angles[i]) for i, j in enumerate(joint_names)}
            velocities: Dict[str, float] = {}
            otg_state = getattr(self, "_otg_state", None)
            vel_state = otg_state.get(arm) if otg_state else None
            if vel_state is not None:
                for i in range(6):
                    velocities[joint_names[i]] = float(vel_state[1][i])
            torques: Dict[str, float] = {}
            if self.config.rebot_gravity_ff:
                kin = self._ensure_rebot_kin()
                g = kin.gravity(np.deg2rad(np.asarray(angles[:6], dtype=float)))
                for i in range(6):
                    torques[joint_names[i]] = float(g[i])
            robot.send_action_mit(positions, velocities, torques)
        else:
            action_dict = {
                f"{j}.pos": float(angles[i]) for i, j in enumerate(joint_names)
            }
            robot.send_action(action_dict)

    def send_command(self) -> bool:
        """Send current joint angles to robot using dictionary format."""
        if not self.is_connected or not self.is_engaged:
            return False
        
        current_time = time.time()
        if current_time - self.last_send_time < self.config.send_interval:
            return True  # Don't send too frequently
        
        try:
            # Send commands with dictionary format - no joint direction mapping
            success = True
            
            joint_names = self.config.joint_names

            # Send left arm command
            if self.left_robot and self.left_arm_connected:
                try:
                    self._send_to_driver(self.left_robot, "left")
                except Exception as e:
                    logger.error(f"Error sending left arm command: {e}")
                    self.left_arm_errors += 1
                    if self.left_arm_errors > self.max_arm_errors:
                        self.left_arm_connected = False
                        logger.error("❌ Left arm disconnected due to repeated errors")
                    success = False

            # Send right arm command
            if self.right_robot and self.right_arm_connected:
                try:
                    self._send_to_driver(self.right_robot, "right")
                except Exception as e:
                    logger.error(f"Error sending right arm command: {e}")
                    self.right_arm_errors += 1
                    if self.right_arm_errors > self.max_arm_errors:
                        self.right_arm_connected = False
                        logger.error("❌ Right arm disconnected due to repeated errors")
                    success = False
            
            self.last_send_time = current_time
            return success
            
        except Exception as e:
            logger.error(f"Error sending robot command: {e}")
            self.general_errors += 1
            if self.general_errors > self.max_general_errors:
                self.is_connected = False
                logger.error("❌ Robot interface disconnected due to repeated errors")
            return False
    
    def set_gripper(self, arm: str, closed: bool):
        """Set gripper state for specified arm."""
        angle = GRIPPER_CLOSED_ANGLE if closed else GRIPPER_OPEN_ANGLE
        gi = self.config.gripper_index
        if arm == "left":
            self.left_arm_angles[gi] = angle
        elif arm == "right":
            self.right_arm_angles[gi] = angle
        else:
            raise ValueError(f"Invalid arm: {arm}")
    
    def get_arm_angles(self, arm: str) -> np.ndarray:
        """Get current joint angles for specified arm."""
        if arm == "left":
            angles = self.left_arm_angles.copy()
        elif arm == "right":
            angles = self.right_arm_angles.copy()
        else:
            raise ValueError(f"Invalid arm: {arm}")
        
        return angles
    
    def get_arm_angles_for_visualization(self, arm: str) -> np.ndarray:
        """Get current joint angles for specified arm, for PyBullet visualization."""
        # Return raw angles without any correction for proper diagnosis
        return self.get_arm_angles(arm)
    
    def get_actual_arm_angles(self, arm: str) -> np.ndarray:
        """Get actual joint angles from robot hardware (not commanded angles)."""
        joint_names = self.config.joint_names
        try:
            robot = self.left_robot if arm == "left" else self.right_robot
            connected = self.left_arm_connected if arm == "left" else self.right_arm_connected
            if robot and connected:
                observation = robot.get_observation()
                if observation:
                    return np.array([observation[f"{j}.pos"] for j in joint_names])
        except Exception as e:
            logger.debug(f"Error reading actual arm angles for {arm}: {e}")

        return self.get_arm_angles(arm)
    
    def return_to_initial_position(self):
        """Return both arms to initial position."""
        logger.info("⏪ Returning robot to initial position...")
        
        try:
            # Set initial positions - no direction mapping
            self.left_arm_angles = self.initial_left_arm.copy()
            self.right_arm_angles = self.initial_right_arm.copy()
            
            # Send commands for a few iterations to ensure movement
            for i in range(10):
                self.send_command()
                time.sleep(0.1)
                
            logger.info("✅ Robot returned to initial position")
        except Exception as e:
            logger.error(f"Error returning to initial position: {e}")
    
    def disable_torque(self, arm: str = None):
        """Disable torque on robot joints.

        Args:
            arm: 'left', 'right', or None for both arms
        """
        if not self.is_connected:
            return

        try:
            if arm is None or arm == "left":
                if self.left_robot and self.left_arm_connected:
                    logger.info("Disabling torque on LEFT arm...")
                    self.left_robot.bus.disable_torque()

            if arm is None or arm == "right":
                if self.right_robot and self.right_arm_connected:
                    logger.info("Disabling torque on RIGHT arm...")
                    self.right_robot.bus.disable_torque()

        except Exception as e:
            logger.error(f"Error disabling torque: {e}")
    
    def disconnect(self):
        """Disconnect from robot hardware."""
        if not self.is_connected:
            return
        
        logger.info("Disconnecting from robot...")
        
        # Return to initial positions if engaged
        if self.is_engaged:
            try:
                self.return_to_initial_position()
            except Exception as e:
                logger.error(f"Error returning to initial position: {e}")
        
        # Disconnect both arms
        if self.left_robot:
            try:
                self.left_robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting left arm: {e}")
            self.left_robot = None
            
        if self.right_robot:
            try:
                self.right_robot.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting right arm: {e}")
            self.right_robot = None
        
        self.is_connected = False
        self.is_engaged = False
        self.left_arm_connected = False
        self.right_arm_connected = False
        logger.info("🔌 Robot disconnected")
    
    def get_arm_connection_status(self, arm: str) -> bool:
        """Get connection status for specific arm based on device file existence."""
        if not self.config.arm_enabled.get(arm, True):
            return False

        # Only check device file existence - ignore overall robot connection status
        if arm == "left":
            device_path = self.config.follower_ports["left"]
            return os.path.exists(device_path)
        elif arm == "right":
            device_path = self.config.follower_ports["right"] 
            return os.path.exists(device_path)
        else:
            return False

    def update_arm_connection_status(self):
        """Update individual arm connection status based on device file existence."""
        if self.is_connected:
            self.left_arm_connected = self.config.arm_enabled.get("left", True) and os.path.exists(self.config.follower_ports["left"])
            self.right_arm_connected = self.config.arm_enabled.get("right", True) and os.path.exists(self.config.follower_ports["right"])
    
    @property
    def status(self) -> Dict:
        """Get robot status information."""
        return {
            "connected": self.is_connected,
            "left_arm_connected": self.left_arm_connected,
            "right_arm_connected": self.right_arm_connected,
            "left_arm_angles": self.left_arm_angles.tolist(),
            "right_arm_angles": self.right_arm_angles.tolist(),
            "joint_limits_min": self.joint_limits_min_deg.tolist(),
            "joint_limits_max": self.joint_limits_max_deg.tolist(),
        } 
