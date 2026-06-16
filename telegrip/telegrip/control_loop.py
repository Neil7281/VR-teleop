"""
Main control loop for the teleoperation system.
Consumes control goals from the command queue and executes them via the robot interface.
"""

import asyncio
import numpy as np
import logging
import time
import queue  # Add import for thread-safe queue
from typing import Dict, Optional

from scipy.spatial.transform import Rotation as R

from .config import TelegripConfig
from .core.robot_interface import RobotInterface
# PyBulletVisualizer will be imported on demand
from .inputs.base import ControlGoal, ControlMode
# WebKeyboardHandler will be imported on demand to avoid circular imports

logger = logging.getLogger(__name__)

# VR world frame -> robot base frame basis change (same axis convention as
# vr_to_robot_coordinates: robot_x=vr_x, robot_y=-vr_z, robot_z=vr_y).
# A rotation expressed in VR coords R_vr maps to robot coords as M @ R_vr @ M.T.
_VR_TO_ROBOT_BASIS = np.array([
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
])


def _vr_delta_quat_to_robot_rotation(delta_quat_xyzw: np.ndarray) -> np.ndarray:
    """Map a relative controller rotation (VR world frame quaternion) to a
    rotation matrix in the robot base frame."""
    r_vr = R.from_quat(np.asarray(delta_quat_xyzw, dtype=float)).as_matrix()
    return _VR_TO_ROBOT_BASIS @ r_vr @ _VR_TO_ROBOT_BASIS.T


class ArmState:
    """State tracking for a single robot arm."""

    def __init__(self, arm_name: str):
        self.arm_name = arm_name
        self.mode = ControlMode.IDLE
        self.target_position = None
        self.goal_position = None
        self.origin_position = None
        self.origin_wrist_roll_angle = 0.0
        self.origin_wrist_flex_angle = 0.0
        self.origin_wrist_yaw_angle = 0.0
        self.current_wrist_roll = 0.0
        self.current_wrist_flex = 0.0
        self.current_wrist_yaw = 0.0   # reBot only; ignored for SO100

        # reBot 6-DOF pose control: EE orientation at grip origin and the
        # current target orientation (3x3 rotation matrices, robot base frame).
        self.origin_ee_rotation = None
        self.current_target_rotation = None

        # Input watchdog: last time a fresh motion target arrived for this arm.
        self.last_input_time = time.time()
        self._stale_logged = False

    def reset(self):
        """Reset arm state to idle."""
        self.mode = ControlMode.IDLE
        self.target_position = None
        self.goal_position = None
        self.origin_position = None
        self.origin_wrist_roll_angle = 0.0
        self.origin_wrist_flex_angle = 0.0
        self.origin_wrist_yaw_angle = 0.0
        self.origin_ee_rotation = None
        self.current_target_rotation = None


class ControlLoop:
    """Main control loop that processes command queue and controls robot."""
    
    def __init__(self, command_queue: asyncio.Queue, config: TelegripConfig, control_commands_queue: Optional[queue.Queue] = None):
        self.command_queue = command_queue
        self.control_commands_queue = control_commands_queue
        self.config = config
        
        # Components
        self.robot_interface = None
        self.visualizer = None
        self.web_keyboard_handler = None  # Reference to web-based keyboard handler
        
        # Arm states
        self.left_arm = ArmState("left")
        self.right_arm = ArmState("right")
        
        # Control timing
        self.last_log_time = 0
        self.log_interval = 1.0  # Log status every second
        # Throttle visualization so GUI/PyBullet work doesn't steal cycles from
        # the high-rate command path (commands stay at send_interval).
        self.last_viz_time = 0
        self.viz_interval = 0.05  # ~20 Hz visualization
        # Input watchdog (§9): stale input → stop advancing (freeze) and warn.
        self.input_freeze_timeout = 0.15  # s: drop buffered path, hold
        self.input_safe_timeout = 1.0     # s: warn (SAFE hold)
        
        # Debug flags
        self._queue_debug_logged = False
        self._process_debug_logged = False
        
        self.is_running = False
    
    def setup(self) -> bool:
        """Setup robot interface and visualizer."""
        success = True
        setup_errors = []
        
        # Setup robot interface
        try:
            self.robot_interface = RobotInterface(self.config)
            if not self.robot_interface.connect():
                error_msg = "Robot interface failed to connect"
                logger.error(error_msg)
                setup_errors.append(error_msg)
                if self.config.enable_robot:
                    success = False
        except Exception as e:
            error_msg = f"Robot interface setup failed with exception: {e}"
            logger.error(error_msg)
            setup_errors.append(error_msg)
            if self.config.enable_robot:
                success = False
        
        # Setup PyBullet simulation, IK and visualizer
        if self.config.enable_pybullet:
            try:
                # Import PyBulletVisualizer on demand
                from .core.visualizer import PyBulletVisualizer
                
                self.visualizer = PyBulletVisualizer(
                    self.config.get_absolute_urdf_path(), 
                    use_gui=self.config.enable_pybullet_gui,
                    log_level=self.config.log_level
                )
                if not self.visualizer.setup():
                    error_msg = "PyBullet visualizer setup failed"
                    logger.error(error_msg)
                    setup_errors.append(error_msg)
                    self.visualizer = None
                else:
                    # Connect kinematics to robot interface
                    joint_limits_min, joint_limits_max = self.visualizer.get_joint_limits
                    self.robot_interface.setup_kinematics(
                        self.visualizer.physics_client,
                        self.visualizer.robot_ids,  # Pass both robot instances
                        self.visualizer.joint_indices,  # Pass both joint index mappings
                        self.visualizer.end_effector_link_indices,  # Pass both end effector indices
                        joint_limits_min,
                        joint_limits_max
                    )
            except Exception as e:
                error_msg = f"PyBullet visualizer setup failed with exception: {e}"
                logger.error(error_msg)
                setup_errors.append(error_msg)
                self.visualizer = None
        
        # Report all setup issues
        if setup_errors:
            logger.error("Setup failed with the following errors:")
            for i, error in enumerate(setup_errors, 1):
                logger.error(f"  {i}. {error}")
        
        # Set robot interface on web keyboard handler so it can get current positions
        if self.web_keyboard_handler and self.robot_interface:
            self.web_keyboard_handler.set_robot_interface(self.robot_interface)
            logger.info("Set robot interface on web keyboard handler")

        return success
    
    async def start(self):
        """Start the control loop."""
        if not self.setup():
            logger.error("Control loop setup failed")
            return
        
        self.is_running = True
        logger.info("Control loop started")
        
        # Initialize arm states with current robot positions
        self._initialize_arm_states()
        
        # Main control loop
        while self.is_running:
            try:
                # Process command queue
                await self._process_commands()
                
                # Input watchdog before commanding (stale input → freeze)
                self._check_input_watchdog()

                # Update robot (with error resilience)
                self._update_robot_safely()
                
                # Update visualization (throttled — keeps the command path fast)
                if self.visualizer:
                    now = time.time()
                    if now - self.last_viz_time >= self.viz_interval:
                        self.last_viz_time = now
                        self._update_visualization()
                
                # Periodic logging
                self._periodic_logging()
                
                # Control rate
                await asyncio.sleep(self.config.send_interval)
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                await asyncio.sleep(0.1)
        
        logger.info("Control loop stopped")
    
    async def stop(self):
        """Stop the control loop."""
        self.is_running = False

        # Cleanup - disengage robot first (returns to home and disables torque)
        if self.robot_interface:
            if self.robot_interface.is_engaged:
                logger.info("🛑 Disengaging robot before shutdown...")
                self.robot_interface.disengage()
            self.robot_interface.disconnect()

        if self.visualizer:
            self.visualizer.disconnect()
    
    def _initialize_arm_states(self):
        """Initialize arm states with current robot positions."""
        if self.robot_interface:
            # Get current end effector positions
            left_pos = self.robot_interface.get_current_end_effector_position("left")
            right_pos = self.robot_interface.get_current_end_effector_position("right")
            
            # Initialize target positions to current positions (ensure deep copies)
            self.left_arm.target_position = left_pos.copy()
            self.left_arm.goal_position = left_pos.copy()
            self.right_arm.target_position = right_pos.copy()
            self.right_arm.goal_position = right_pos.copy()
            
            # Get current wrist angles
            left_angles = self.robot_interface.get_arm_angles("left")
            right_angles = self.robot_interface.get_arm_angles("right")

            wfi = self.config.wrist_flex_index
            wri = self.config.wrist_roll_index
            wyi = self.config.wrist_yaw_index

            self.left_arm.current_wrist_roll = left_angles[wri]
            self.right_arm.current_wrist_roll = right_angles[wri]
            self.left_arm.current_wrist_flex = left_angles[wfi]
            self.right_arm.current_wrist_flex = right_angles[wfi]
            if wyi is not None:
                self.left_arm.current_wrist_yaw = left_angles[wyi]
                self.right_arm.current_wrist_yaw = right_angles[wyi]
            
            logger.info(f"Initialized left arm at position: {left_pos.round(3)}")
            logger.info(f"Initialized right arm at position: {right_pos.round(3)}")
    
    async def _process_commands(self):
        """Process commands from the command queue."""
        try:
            # Process regular control goals
            while not self.command_queue.empty():
                goal = self.command_queue.get_nowait()
                await self._execute_goal(goal)
        except Exception as e:
            logger.error(f"Error processing commands: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def _handle_command(self, command):
        """Handle individual commands."""
        action = command.get('action', '')
        logger.info(f"🔌 Processing control command: {action}")
        
        if action == 'enable_keyboard':
            if self.web_keyboard_handler:
                await self.web_keyboard_handler.start()
                logger.info("🎮 Keyboard control ENABLED via API")
        elif action == 'disable_keyboard':
            if self.web_keyboard_handler:
                await self.web_keyboard_handler.stop()
                logger.info("🎮 Keyboard control DISABLED via API")
        elif action == 'web_keypress':
            # Handle individual keypress events from web interface
            key = command.get('key')
            event = command.get('event')  # 'press' or 'release'

            if self.web_keyboard_handler and self.web_keyboard_handler.is_enabled:
                logger.debug(f"🌐 Processing web keypress: {key}_{event}")
                if event == 'press':
                    self.web_keyboard_handler.on_key_press(key)
                elif event == 'release':
                    self.web_keyboard_handler.on_key_release(key)
            else:
                logger.warning("🎮 Web keyboard handler not enabled")
        elif action == 'robot_connect':
            logger.info("🔌 Processing robot_connect command")
            if self.robot_interface and self.robot_interface.is_connected:
                logger.info(f"🔌 Robot interface available and connected: {self.robot_interface.is_connected}")
                success = self.robot_interface.engage()
                if success:
                    logger.info("🔌 Robot motors ENGAGED via API")
                    # No need to sync keyboard targets - unified system handles this automatically
                else:
                    logger.error("❌ Failed to engage robot motors")
            else:
                logger.warning(f"Cannot engage robot: interface={self.robot_interface is not None}, connected={self.robot_interface.is_connected if self.robot_interface else False}")
        elif action == 'robot_disconnect':
            logger.info("🔌 Processing robot_disconnect command")
            if self.robot_interface:
                logger.info(f"🔌 Robot interface available")
                success = self.robot_interface.disengage()
                if success:
                    logger.info("🔌 Robot motors DISENGAGED via API")
                    # Reset arm states to IDLE when robot is disengaged
                    self.left_arm.reset()
                    self.right_arm.reset()
                    logger.info("🔓 Both arms: Position control DEACTIVATED after robot disconnect")
                    
                    # Hide visualization markers
                    if self.visualizer:
                        for arm in ["left", "right"]:
                            self.visualizer.hide_marker(f"{arm}_goal")
                            self.visualizer.hide_frame(f"{arm}_goal_frame")
                            self.visualizer.hide_marker(f"{arm}_target")
                            self.visualizer.hide_frame(f"{arm}_target_frame")
                else:
                    logger.error("❌ Failed to disengage robot motors")
            else:
                logger.warning("Cannot disengage robot: no robot interface")
        else:
            logger.warning(f"Unknown command: {action}")

    def _capture_rebot_origin(self, arm_state: ArmState, arm: str):
        """reBot: capture EE position + orientation at grip origin via official FK.

        Overrides origin_position with the Pinocchio FK position so the origin,
        target and IK all use one consistent kinematic model.
        """
        if not self.config.is_rebot or not self.robot_interface:
            return
        try:
            # Re-anchor the model to the real arm before capturing the origin.
            self.robot_interface.sync_commanded_to_actual(arm)
            pos, rot = self.robot_interface.get_current_end_effector_pose(arm)
            arm_state.origin_position = pos.copy()
            arm_state.target_position = pos.copy()
            arm_state.goal_position = pos.copy()
            arm_state.origin_ee_rotation = rot.copy()
            arm_state.current_target_rotation = rot.copy()
            # Start trajectory servoing from the current pose (zero initial step).
            self.robot_interface.anchor_rebot_tracking(arm)
        except Exception as e:
            logger.error(f"reBot origin pose capture failed for {arm}: {e}")

    async def _execute_goal(self, goal: ControlGoal):
        """Execute a control goal."""
        arm_state = self.left_arm if goal.arm == "left" else self.right_arm
        
        wfi = self.config.wrist_flex_index
        wri = self.config.wrist_roll_index
        wyi = self.config.wrist_yaw_index

        # Handle special reset signal from keyboard idle timeout
        if (goal.metadata and goal.metadata.get("reset_target_to_current", False)):
            if self.robot_interface and arm_state.mode == ControlMode.POSITION_CONTROL:
                current_position = self.robot_interface.get_current_end_effector_position(goal.arm)
                current_angles = self.robot_interface.get_arm_angles(goal.arm)

                arm_state.target_position = current_position.copy()
                arm_state.goal_position = current_position.copy()
                arm_state.origin_position = current_position.copy()
                arm_state.current_wrist_roll = current_angles[wri]
                arm_state.current_wrist_flex = current_angles[wfi]
                arm_state.origin_wrist_roll_angle = current_angles[wri]
                arm_state.origin_wrist_flex_angle = current_angles[wfi]
                if wyi is not None:
                    arm_state.current_wrist_yaw = current_angles[wyi]
                    arm_state.origin_wrist_yaw_angle = current_angles[wyi]

                self._capture_rebot_origin(arm_state, goal.arm)

                logger.info(f"🔄 {goal.arm.upper()} arm: Target position reset to current robot position (idle timeout)")
            return
        
        # Handle mode changes (only if mode is specified)
        if goal.mode is not None and goal.mode != arm_state.mode:
            if goal.mode == ControlMode.POSITION_CONTROL:
                arm_state.mode = ControlMode.POSITION_CONTROL

                if self.robot_interface:
                    current_position = self.robot_interface.get_current_end_effector_position(goal.arm)
                    current_angles = self.robot_interface.get_arm_angles(goal.arm)

                    arm_state.target_position = current_position.copy()
                    arm_state.goal_position = current_position.copy()
                    arm_state.origin_position = current_position.copy()
                    arm_state.current_wrist_roll = current_angles[wri]
                    arm_state.current_wrist_flex = current_angles[wfi]
                    arm_state.origin_wrist_roll_angle = current_angles[wri]
                    arm_state.origin_wrist_flex_angle = current_angles[wfi]
                    if wyi is not None:
                        arm_state.current_wrist_yaw = current_angles[wyi]
                        arm_state.origin_wrist_yaw_angle = current_angles[wyi]

                    self._capture_rebot_origin(arm_state, goal.arm)

                logger.info(f"🔒 {goal.arm.upper()} arm: Position control ACTIVATED (target reset to current position)")
                
            elif goal.mode == ControlMode.IDLE:
                # Deactivate position control
                arm_state.reset()
                
                # Hide visualization markers
                if self.visualizer:
                    self.visualizer.hide_marker(f"{goal.arm}_goal")
                    self.visualizer.hide_frame(f"{goal.arm}_goal_frame")
                
                logger.info(f"🔓 {goal.arm.upper()} arm: Position control DEACTIVATED")
        
        # Handle position control - both VR and keyboard now work the same way (absolute offset from origin)
        if goal.target_position is not None and arm_state.mode == ControlMode.POSITION_CONTROL:
            # Fresh motion input → pet the watchdog.
            arm_state.last_input_time = time.time()
            arm_state._stale_logged = False
            if goal.metadata and goal.metadata.get("relative_position", False):
                # Both VR and keyboard send absolute offset from robot origin position
                if arm_state.origin_position is not None:
                    arm_state.target_position = arm_state.origin_position + goal.target_position
                    arm_state.goal_position = arm_state.target_position.copy()
                else:
                    # No origin set yet, use current position as base
                    if self.robot_interface:
                        current_position = self.robot_interface.get_current_end_effector_position(goal.arm)
                        arm_state.target_position = current_position + goal.target_position
                        arm_state.goal_position = arm_state.target_position.copy()
            else:
                # Absolute position (legacy - should not be used anymore)
                arm_state.target_position = goal.target_position.copy()
                arm_state.goal_position = goal.target_position.copy()
            
            # Handle orientation.
            relative = goal.metadata and goal.metadata.get("relative_position", False)

            if self.config.is_rebot:
                # reBot: full 6-DOF — apply the relative controller rotation to the
                # EE orientation captured at grip origin. With orientation disabled
                # (safe bring-up), the target rotation stays at the origin (held).
                if (self.config.rebot_orientation_enabled and
                        goal.delta_rotation_quat is not None and
                        arm_state.origin_ee_rotation is not None):
                    delta_robot = _vr_delta_quat_to_robot_rotation(goal.delta_rotation_quat)
                    # Sensitivity: scale the rotation magnitude (wrist gain).
                    scale = self.config.rebot_orientation_scale
                    if scale != 1.0:
                        rotvec = R.from_matrix(delta_robot).as_rotvec() * scale
                        delta_robot = R.from_rotvec(rotvec).as_matrix()
                    arm_state.current_target_rotation = delta_robot @ arm_state.origin_ee_rotation
            else:
                # SO100: per-axis wrist angles (absolute offset from origin).
                if goal.wrist_roll_deg is not None:
                    arm_state.current_wrist_roll = (
                        arm_state.origin_wrist_roll_angle + goal.wrist_roll_deg
                        if relative else goal.wrist_roll_deg
                    )

                if goal.wrist_flex_deg is not None:
                    arm_state.current_wrist_flex = (
                        arm_state.origin_wrist_flex_angle + goal.wrist_flex_deg
                        if relative else goal.wrist_flex_deg
                    )

                if goal.wrist_yaw_deg is not None and wyi is not None:
                    arm_state.current_wrist_yaw = (
                        arm_state.origin_wrist_yaw_angle + goal.wrist_yaw_deg
                        if relative else goal.wrist_yaw_deg
                    )
        
        # Handle gripper control (independent of mode)
        if goal.gripper_closed is not None and self.robot_interface:
            self.robot_interface.set_gripper(goal.arm, goal.gripper_closed)
    
    def _check_input_watchdog(self):
        """Freeze an engaged arm if its motion input has gone stale.

        On stale input (e.g. headset sleep / Wi-Fi drop), drop any buffered path
        so the arm stops advancing and holds at its current pose — it never
        keeps executing queued (now-stale) motion. Pure hold, no return-to-home.
        """
        if not self.robot_interface or not self.robot_interface.is_engaged:
            return
        now = time.time()
        for arm_state, arm in ((self.left_arm, "left"), (self.right_arm, "right")):
            if arm_state.mode != ControlMode.POSITION_CONTROL:
                continue
            stale = now - arm_state.last_input_time
            if stale > self.input_freeze_timeout:
                self.robot_interface.freeze_rebot(arm)
                if stale > self.input_safe_timeout and not arm_state._stale_logged:
                    logger.warning(
                        f"⚠️  {arm.upper()} input stale {stale:.1f}s — holding pose (SAFE)."
                    )
                    arm_state._stale_logged = True

    def _update_robot_safely(self):
        """Update robot with current control goals (with error handling)."""
        if not self.robot_interface:
            return
        
        try:
            self._update_robot()
        except Exception as e:
            logger.error(f"Error updating robot: {e}")
            # Don't shutdown, just continue - robot interface will handle connection issues
    
    def _command_arm(self, arm_state: ArmState, arm: str):
        """Resolve one arm's target into joint angles and store them.

        reBot uses the official 6-DOF pose IK (position + orientation); SO100
        uses the position-only IK + per-axis wrist control.
        """
        if (arm_state.mode != ControlMode.POSITION_CONTROL or
                arm_state.target_position is None or
                not self.robot_interface.get_arm_connection_status(arm)):
            return

        if self.config.is_rebot:
            if arm_state.current_target_rotation is not None:
                # Path following: sample the hand pose into a min-distance
                # waypoint queue, then trace that polyline at a bounded Cartesian
                # speed so the EE reproduces the hand's trajectory.
                dt = self.config.send_interval
                max_lin = self.config.rebot_max_lin_vel_m_s * dt
                max_ang = np.radians(self.config.rebot_max_ang_vel_deg_s) * dt
                self.robot_interface.add_waypoint(
                    arm, arm_state.target_position, arm_state.current_target_rotation,
                    self.config.rebot_waypoint_min_dist_m,
                    np.radians(self.config.rebot_waypoint_min_ang_deg),
                    self.config.rebot_path_max_waypoints,
                )
                self.robot_interface.track_path(arm, max_lin, max_ang)
            return

        gi = self.config.gripper_index
        ik_solution = self.robot_interface.solve_ik(arm, arm_state.target_position)
        current_gripper = self.robot_interface.get_arm_angles(arm)[gi]
        self.robot_interface.update_arm_angles(
            arm, ik_solution,
            arm_state.current_wrist_flex,
            arm_state.current_wrist_roll,
            current_gripper,
            wrist_yaw=arm_state.current_wrist_yaw,
        )

    def _update_robot(self):
        """Update robot with current control goals."""
        if not self.robot_interface:
            return

        self._command_arm(self.left_arm, "left")
        self._command_arm(self.right_arm, "right")

        # Send commands to robot
        if self.robot_interface.is_connected and self.robot_interface.is_engaged:
            self.robot_interface.send_command()
    
    def _update_visualization(self):
        """Update PyBullet visualization."""
        if not self.visualizer:
            return
        
        # Use COMMANDED angles for display (not a hardware read). Reading actual
        # feedback here every cycle polls the Damiao bus and contends with motor
        # command sending on the same link, causing jerky/laggy motion. Commanded
        # angles track the real arm closely enough for visualization.
        left_angles = self.robot_interface.get_arm_angles("left")
        right_angles = self.robot_interface.get_arm_angles("right")

        self.visualizer.update_robot_pose(left_angles, 'left')
        self.visualizer.update_robot_pose(right_angles, 'right')
        
        # Update visualization markers
        if self.left_arm.mode == ControlMode.POSITION_CONTROL:
            if self.left_arm.target_position is not None:
                # Show current end effector position
                current_pos = self.robot_interface.get_current_end_effector_position("left")
                self.visualizer.update_marker_position("left_target", current_pos)
                self.visualizer.update_coordinate_frame("left_target_frame", current_pos)
            
            if self.left_arm.goal_position is not None:
                # Show goal position
                self.visualizer.update_marker_position("left_goal", self.left_arm.goal_position)
                self.visualizer.update_coordinate_frame("left_goal_frame", self.left_arm.goal_position)
        else:
            # Hide markers when not in position control
            self.visualizer.hide_marker("left_target")
            self.visualizer.hide_marker("left_goal")
            self.visualizer.hide_frame("left_target_frame")
            self.visualizer.hide_frame("left_goal_frame")
        
        if self.right_arm.mode == ControlMode.POSITION_CONTROL:
            if self.right_arm.target_position is not None:
                # Show current end effector position
                current_pos = self.robot_interface.get_current_end_effector_position("right")
                self.visualizer.update_marker_position("right_target", current_pos)
                self.visualizer.update_coordinate_frame("right_target_frame", current_pos)
            
            if self.right_arm.goal_position is not None:
                # Show goal position
                self.visualizer.update_marker_position("right_goal", self.right_arm.goal_position)
                self.visualizer.update_coordinate_frame("right_goal_frame", self.right_arm.goal_position)
        else:
            # Hide markers when not in position control
            self.visualizer.hide_marker("right_target")
            self.visualizer.hide_marker("right_goal")
            self.visualizer.hide_frame("right_target_frame")
            self.visualizer.hide_frame("right_goal_frame")
        
        # Step simulation
        self.visualizer.step_simulation()
    
    def _periodic_logging(self):
        """Log status information periodically."""
        self._loop_cycles = getattr(self, "_loop_cycles", 0) + 1
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            elapsed = current_time - self.last_log_time
            hz = self._loop_cycles / elapsed if elapsed > 0 else 0.0
            self._loop_cycles = 0
            self.last_log_time = current_time

            active_arms = []
            if self.left_arm.mode == ControlMode.POSITION_CONTROL:
                active_arms.append("LEFT")
            if self.right_arm.mode == ControlMode.POSITION_CONTROL:
                active_arms.append("RIGHT")

            if active_arms and self.robot_interface:
                right_angles = self.robot_interface.get_arm_angles("right")
                logger.info(
                    f"🤖 Active: {', '.join(active_arms)} | loop {hz:.0f} Hz "
                    f"(target {1.0/self.config.send_interval:.0f}) | Right: {right_angles.round(1)}"
                )
    
    @property
    def status(self) -> Dict:
        """Get current control loop status."""
        return {
            "running": self.is_running,
            "left_arm_mode": self.left_arm.mode.value,
            "right_arm_mode": self.right_arm.mode.value,
            "robot_connected": self.robot_interface.is_connected if self.robot_interface else False,
            "left_arm_connected": self.robot_interface.get_arm_connection_status("left") if self.robot_interface else False,
            "right_arm_connected": self.robot_interface.get_arm_connection_status("right") if self.robot_interface else False,
            "visualizer_connected": self.visualizer.is_connected if self.visualizer else False,
        } 
