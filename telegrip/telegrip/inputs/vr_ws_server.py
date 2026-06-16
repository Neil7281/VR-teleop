"""
VR WebSocket server for receiving controller data from web browsers.
Adapted from the original vr_robot_teleop.py script.
"""

import asyncio
import json
import ssl
import websockets
import numpy as np
import math
import time
import logging
from typing import Dict, Optional, Set
from scipy.spatial.transform import Rotation as R

from .base import BaseInputProvider, ControlGoal, ControlMode
from ..config import TelegripConfig
from ..core.kinematics import compute_relative_position

logger = logging.getLogger(__name__)


class _EMAVec3:
    """Exponential moving average filter for a 3-D position vector."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, raw: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = raw.copy()
        else:
            self._state = self.alpha * raw + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self, initial: Optional[np.ndarray] = None) -> None:
        self._state = initial.copy() if initial is not None else None


class _EMAScalar:
    """Exponential moving average filter for a scalar (angle)."""

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self._state: float = 0.0

    def update(self, raw: float) -> float:
        self._state = self.alpha * raw + (1.0 - self.alpha) * self._state
        return self._state

    def reset(self) -> None:
        self._state = 0.0


class _EMAQuat:
    """EMA filter for a unit quaternion [x,y,z,w] via normalized lerp (nlerp).

    Hemisphere-aligned so the shortest path is taken; good enough for the small
    per-frame orientation changes in teleop and avoids the jitter of feeding raw
    controller orientation straight into IK.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        raw = raw / (np.linalg.norm(raw) + 1e-12)
        if self._state is None:
            self._state = raw.copy()
            return self._state.copy()
        # Align hemispheres so we interpolate the short way.
        if np.dot(self._state, raw) < 0.0:
            raw = -raw
        s = self.alpha * raw + (1.0 - self.alpha) * self._state
        self._state = s / (np.linalg.norm(s) + 1e-12)
        return self._state.copy()

    def reset(self, initial: Optional[np.ndarray] = None) -> None:
        self._state = (
            np.asarray(initial, dtype=float).copy() if initial is not None else None
        )


class _OneEuro:
    """One Euro filter for a scalar (Casiez et al.). Adaptive low-pass: heavy
    smoothing when slow, light when fast. Uses real timestamps."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.02, dcutoff: float = 1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev: Optional[float] = None
        self._dx_prev = 0.0
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def update(self, x: float, t: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x
        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3
        self._t_prev = t
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class _OneEuroVec3:
    """One Euro filter for a 3-D vector (same interface as _EMAVec3)."""

    def __init__(self, mincutoff: float = 1.0, beta: float = 0.02):
        self._f = [_OneEuro(mincutoff, beta) for _ in range(3)]

    def update(self, raw: np.ndarray) -> np.ndarray:
        t = time.monotonic()
        return np.array([self._f[i].update(float(raw[i]), t) for i in range(3)])

    def reset(self, initial: Optional[np.ndarray] = None) -> None:
        for f in self._f:
            f.reset()
        if initial is not None:
            t = time.monotonic()
            for i, f in enumerate(self._f):
                f.update(float(initial[i]), t)


class VRControllerState:
    """State tracking for a VR controller."""

    def __init__(self, hand: str, oneeuro_mincutoff: float = 1.0,
                 oneeuro_beta: float = 0.02, quat_alpha: float = 0.3):
        self.hand = hand
        self.grip_active = False
        self.trigger_active = False

        # Position tracking for relative movement
        self.origin_position = None
        self.origin_rotation = None

        # Quaternion-based rotation tracking
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None

        # Rotation tracking for wrist control
        self.z_axis_rotation = 0.0  # wrist_roll
        self.x_axis_rotation = 0.0  # wrist_flex
        self.y_axis_rotation = 0.0  # wrist_yaw (reBot only)

        self.current_position = None
        self.origin_wrist_angle = 0.0

        # ── Smoothing filters ──────────────────────────────────────────
        # alpha=0.4 → ~17 ms effective lag at 90 Hz; smooths hand tremor
        # without feeling sluggish.
        # One Euro adaptive filter on position (precision when slow, no lag fast).
        self.pos_filter = _OneEuroVec3(mincutoff=oneeuro_mincutoff, beta=oneeuro_beta)
        # Wrist angles are noisier; slightly heavier filter.
        self.roll_filter  = _EMAScalar(alpha=0.35)
        self.flex_filter  = _EMAScalar(alpha=0.35)
        self.yaw_filter   = _EMAScalar(alpha=0.35)
        # reBot full-orientation smoothing (quaternion).
        self.quat_filter  = _EMAQuat(alpha=quat_alpha)

    def reset_grip(self):
        """Reset grip state but preserve trigger state."""
        self.grip_active = False
        self.origin_position = None
        self.origin_rotation = None
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None
        self.z_axis_rotation = 0.0
        self.x_axis_rotation = 0.0
        self.y_axis_rotation = 0.0
        # Reset filters so next grip starts clean
        self.pos_filter.reset()
        self.roll_filter.reset()
        self.flex_filter.reset()
        self.yaw_filter.reset()
        self.quat_filter.reset()


class VRWebSocketServer(BaseInputProvider):
    """WebSocket server for VR controller input."""
    
    def __init__(self, command_queue: asyncio.Queue, config: TelegripConfig):
        super().__init__(command_queue)
        self.config = config
        self.clients: Set = set()
        self.server = None
        
        # Controller states (filter strengths are config-tunable for reBot)
        _mc = float(getattr(config, "rebot_oneeuro_mincutoff", 1.0))
        _beta = float(getattr(config, "rebot_oneeuro_beta", 0.02))
        _qa = float(getattr(config, "rebot_orient_filter_alpha", 0.3))
        self.left_controller = VRControllerState(
            "left", oneeuro_mincutoff=_mc, oneeuro_beta=_beta, quat_alpha=_qa)
        self.right_controller = VRControllerState(
            "right", oneeuro_mincutoff=_mc, oneeuro_beta=_beta, quat_alpha=_qa)
        
        # Robot state tracking (for relative position calculation)
        self.left_arm_origin_position = None
        self.right_arm_origin_position = None

    def _get_local_ip(self) -> str:
        """Get the local IP address of this machine."""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            try:
                return socket.gethostbyname(socket.gethostname())
            except Exception:
                return "localhost"

    def setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context for WebSocket server."""
        # Automatically generate SSL certificates if they don't exist
        if not self.config.ssl_files_exist:
            logger.info("SSL certificates not found for WebSocket server, attempting to generate them...")
            if not self.config.ensure_ssl_certificates():
                logger.error("Failed to generate SSL certificates for WebSocket server")
                return None
        
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            # Get absolute paths for SSL certificates
            cert_path, key_path = self.config.get_absolute_ssl_paths()
            ssl_context.load_cert_chain(certfile=cert_path, keyfile=key_path)
            logger.info("SSL certificate and key loaded successfully for WebSocket server")
            return ssl_context
        except ssl.SSLError as e:
            logger.error(f"Error loading SSL cert/key: {e}")
            return None
    
    async def start(self):
        """Start the WebSocket server."""
        if not self.config.enable_vr:
            logger.info("VR WebSocket server disabled in configuration")
            return
        
        ssl_context = self.setup_ssl()
        if ssl_context is None:
            logger.error("Failed to setup SSL for WebSocket server")
            return
        
        host = self.config.host_ip
        port = self.config.websocket_port
        self._browser_warning_shown = False

        try:
            self.server = await websockets.serve(
                self.websocket_handler,
                host,
                port,
                ssl=ssl_context,
                process_request=self._process_request
            )
            self.is_running = True
            host_display = self._get_local_ip() if host == "0.0.0.0" else host
            logger.info(f"VR WebSocket server running on wss://{host_display}:{port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")

    async def _process_request(self, connection, request):
        """Process incoming requests and detect browser visits to the WebSocket port."""
        # Check if this looks like a browser request (not a proper WebSocket upgrade)
        # In newer websockets versions, request.headers is a Headers object
        headers = request.headers
        connection_header = headers.get("Connection", "")
        upgrade_header = headers.get("Upgrade", "")

        # Proper WebSocket requests have "Upgrade" in Connection header and "websocket" in Upgrade header
        is_websocket_request = (
            "upgrade" in connection_header.lower() and
            "websocket" in upgrade_header.lower()
        )

        if not is_websocket_request:
            # Only show warning once to avoid spam
            if not self._browser_warning_shown:
                self._browser_warning_shown = True
                host_display = self._get_local_ip() if self.config.host_ip == "0.0.0.0" else self.config.host_ip
                print(f"\n⚠️  Someone is trying to open port {self.config.websocket_port} in a browser.")
                print(f"   This port is for VR WebSocket connections only.")
                print(f"   The web UI is at: https://{host_display}:{self.config.https_port}\n")

        # Return None to let websockets library handle the request normally
        # (it will reject non-WebSocket requests with 426 Upgrade Required)
        return None

    async def stop(self):
        """Stop the WebSocket server."""
        self.is_running = False

        # Close all active client connections to unblock websocket_handler
        for client in list(self.clients):
            try:
                await client.close()
            except Exception:
                pass

        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("VR WebSocket server stopped")
    
    async def websocket_handler(self, websocket, path=None):
        """Handle WebSocket connections from VR controllers."""
        client_address = websocket.remote_address
        logger.info(f"VR client connected: {client_address}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_controller_data(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received non-JSON message: {message}")
                except Exception as e:
                    logger.error(f"Error processing VR data: {e}")
        
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"VR client {client_address} disconnected normally")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"VR client {client_address} disconnected with error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error with VR client {client_address}: {e}")
        finally:
            self.clients.discard(websocket)
            # Handle grip releases when client disconnects
            await self.handle_grip_release('left')
            await self.handle_grip_release('right')
            logger.info(f"VR client {client_address} cleanup complete")
    
    async def process_controller_data(self, data: Dict):
        """Process incoming VR controller data."""
        
        # Handle new dual controller format
        if 'leftController' in data and 'rightController' in data:
            left_data = data['leftController']
            right_data = data['rightController']
            
            # Process left controller
            if left_data.get('position') and (left_data.get('gripActive', False) or left_data.get('trigger', 0) > 0.5):
                await self.process_single_controller('left', left_data)
            elif not left_data.get('gripActive', False) and self.left_controller.grip_active:
                await self.handle_grip_release('left')
            
            # Process right controller
            if right_data.get('position') and (right_data.get('gripActive', False) or right_data.get('trigger', 0) > 0.5):
                await self.process_single_controller('right', right_data)
            elif not right_data.get('gripActive', False) and self.right_controller.grip_active:
                await self.handle_grip_release('right')
                
            return
        
        # Handle legacy single controller format
        hand = data.get('hand')
        
        # Handle explicit release messages
        if data.get('gripReleased'):
            await self.handle_grip_release(hand)
            return
        
        if data.get('triggerReleased'):
            await self.handle_trigger_release(hand)
            return
            
        # Process single controller data
        if hand and data.get('position') and (data.get('gripActive', False) or data.get('trigger', 0) > 0.5):
            await self.process_single_controller(hand, data)
    
    async def process_single_controller(self, hand: str, data: Dict):
        """Process data for a single controller."""
        position = data.get('position', {})
        rotation = data.get('rotation', {})
        quaternion = data.get('quaternion', {})  # Get quaternion data directly
        grip_active = data.get('gripActive', False)
        trigger = data.get('trigger', 0)
        
        controller = self.left_controller if hand == 'left' else self.right_controller
        
        # Handle trigger for gripper control
        trigger_active = trigger > 0.5
        if trigger_active != controller.trigger_active:
            controller.trigger_active = trigger_active
            
            # Send gripper control goal - do not specify mode to avoid interfering with position control
            # Reverse behavior: gripper open by default, closes when trigger pressed
            gripper_goal = ControlGoal(
                arm=hand,
                gripper_closed=not trigger_active,  # Inverted: closed when trigger NOT active
                metadata={"source": "vr_trigger"}
            )
            await self.send_goal(gripper_goal)
            
            logger.info(f"🤏 {hand.upper()} gripper {'OPENED' if trigger_active else 'CLOSED'}")
        
        # Handle grip button for arm movement control
        if grip_active:
            if not controller.grip_active:
                # Grip just activated — seed filters at current raw position
                # so the first filtered value equals the origin (delta = 0).
                pos_np = np.array([position['x'], position['y'], position['z']])
                controller.pos_filter.reset(pos_np)
                controller.roll_filter.reset()
                controller.flex_filter.reset()
                controller.yaw_filter.reset()

                controller.grip_active = True
                controller.origin_position = position.copy()
                
                # Use quaternion data directly if available, otherwise fall back to Euler conversion
                if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                    controller.origin_quaternion = np.array([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']])
                    controller.origin_rotation = controller.origin_quaternion  # Store for compatibility
                else:
                    # Fallback to Euler angle conversion
                    controller.origin_quaternion = self.euler_to_quaternion(rotation) if rotation else None
                    controller.origin_rotation = controller.origin_quaternion
                
                controller.accumulated_rotation_quat = controller.origin_quaternion
                # Seed the quaternion filter at the origin so the first delta is 0.
                controller.quat_filter.reset(controller.origin_quaternion)
                controller.z_axis_rotation = 0.0
                controller.x_axis_rotation = 0.0
                
                # Send reset signal to control loop to reset target position to current robot position
                reset_goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,  # Keep in position control
                    target_position=None,  # Special signal
                    metadata={
                        "source": f"vr_grip_reset_{hand}",
                        "reset_target_to_current": True  # Signal to reset target to current position
                    }
                )
                await self.send_goal(reset_goal)
                
                logger.info(f"🔒 {hand.upper()} grip activated - controlling {hand} arm (target reset to current position)")
            
            # Compute target position
            if controller.origin_position:
                # ── Filter raw position to remove hand tremor ──────────
                pos_np = np.array([position['x'], position['y'], position['z']])
                f = controller.pos_filter.update(pos_np)
                filtered_position = {'x': float(f[0]), 'y': float(f[1]), 'z': float(f[2])}

                relative_delta = compute_relative_position(
                    filtered_position,
                    controller.origin_position,
                    self.config.vr_to_robot_scale
                )

                # ── Extract and filter wrist rotations ─────────────────
                # Full relative controller rotation (VR world frame) for reBot
                # 6-DOF pose IK. SO100 ignores this and uses the scalar angles.
                delta_rotation_quat = None
                if controller.origin_quaternion is not None:
                    if quaternion and all(k in quaternion for k in ['x', 'y', 'z', 'w']):
                        current_quat = np.array([
                            quaternion['x'], quaternion['y'],
                            quaternion['z'], quaternion['w'],
                        ])
                        self.update_quaternion_rotation_direct(controller, current_quat)
                        # Smooth the absolute orientation, then take the relative
                        # rotation from the (fixed) grip origin.
                        filtered_quat = controller.quat_filter.update(current_quat)
                        delta_rotation_quat = (
                            R.from_quat(filtered_quat)
                            * R.from_quat(controller.origin_quaternion).inv()
                        ).as_quat()
                    else:
                        self.update_quaternion_rotation(controller, rotation)

                    controller.z_axis_rotation = controller.roll_filter.update(
                        self.extract_roll_from_quaternion(
                            controller.accumulated_rotation_quat, controller.origin_quaternion))
                    controller.x_axis_rotation = controller.flex_filter.update(
                        self.extract_pitch_from_quaternion(
                            controller.accumulated_rotation_quat, controller.origin_quaternion))
                    controller.y_axis_rotation = controller.yaw_filter.update(
                        self.extract_yaw_from_quaternion(
                            controller.accumulated_rotation_quat, controller.origin_quaternion))

                goal = ControlGoal(
                    arm=hand,
                    mode=ControlMode.POSITION_CONTROL,
                    target_position=relative_delta,
                    wrist_roll_deg=-controller.z_axis_rotation,
                    wrist_flex_deg=-controller.x_axis_rotation,
                    wrist_yaw_deg=controller.y_axis_rotation,
                    delta_rotation_quat=delta_rotation_quat,
                    metadata={
                        "source": "vr_grip",
                        "relative_position": True,
                        "origin_position": controller.origin_position.copy()
                    }
                )
                await self.send_goal(goal)
    
    async def handle_grip_release(self, hand: str):
        """Handle grip release for a controller."""
        if hand == 'left':
            controller = self.left_controller
        elif hand == 'right':
            controller = self.right_controller
        else:
            return
        
        if controller.grip_active:
            controller.reset_grip()
            
            # Send idle goal to stop arm control
            goal = ControlGoal(
                arm=hand,
                mode=ControlMode.IDLE,
                metadata={"source": "vr_grip_release"}
            )
            await self.send_goal(goal)
            
            logger.info(f"🔓 {hand.upper()} grip released - arm control stopped")
    
    async def handle_trigger_release(self, hand: str):
        """Handle trigger release for a controller."""
        controller = self.left_controller if hand == 'left' else self.right_controller
        
        if controller.trigger_active:
            controller.trigger_active = False
            
            # Send gripper closed goal - reversed behavior: gripper closes when trigger released
            goal = ControlGoal(
                arm=hand,
                gripper_closed=True,  # Close gripper when trigger released
                metadata={"source": "vr_trigger_release"}
            )
            await self.send_goal(goal)
            
            logger.info(f"🤏 {hand.upper()} gripper CLOSED (trigger released)")
    
    def euler_to_quaternion(self, euler_deg: Dict[str, float]) -> np.ndarray:
        """Convert Euler angles in degrees to quaternion [x, y, z, w]."""
        euler_rad = [math.radians(euler_deg['x']), math.radians(euler_deg['y']), math.radians(euler_deg['z'])]
        rotation = R.from_euler('xyz', euler_rad)
        return rotation.as_quat()
    
    def update_quaternion_rotation(self, controller: VRControllerState, current_euler: dict):
        """Update quaternion-based rotation tracking."""
        if not current_euler:
            return
        
        # Convert current Euler to quaternion
        current_quat = self.euler_to_quaternion(current_euler)
        
        # Store current quaternion for accumulated rotation calculation
        controller.accumulated_rotation_quat = current_quat
    
    def update_quaternion_rotation_direct(self, controller: VRControllerState, current_quat: np.ndarray):
        """Update quaternion-based rotation tracking using quaternion data directly."""
        if current_quat is None:
            return
        
        # Store current quaternion for accumulated rotation calculation
        controller.accumulated_rotation_quat = current_quat
    
    def extract_yaw_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract yaw rotation around Y-axis from relative quaternion — maps to reBot wrist_yaw."""
        if current_quat is None or origin_quat is None:
            return 0.0
        try:
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()
            rotvec = relative_rotation.as_rotvec()
            return np.degrees(rotvec[1])   # Y-component = yaw
        except Exception as e:
            logger.warning(f"Error extracting yaw from quaternion: {e}")
            return 0.0

    def extract_roll_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract roll rotation around Z-axis from relative quaternion rotation."""
        if current_quat is None or origin_quat is None:
            return 0.0
        
        try:
            # Calculate relative rotation quaternion (from origin to current)
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()
            
            # Project the relative rotation onto the Z-axis (roll)
            # Get the rotation vector (axis-angle representation)
            rotvec = relative_rotation.as_rotvec()
            
            # The Z-component of the rotation vector represents rotation around Z-axis (roll)
            z_rotation_rad = rotvec[2]
            z_rotation_deg = -np.degrees(z_rotation_rad)
            
            return z_rotation_deg
        except Exception as e:
            logger.warning(f"Error extracting roll from quaternion: {e}")
            return 0.0
    
    def extract_pitch_from_quaternion(self, current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        """Extract pitch rotation around X-axis from relative quaternion rotation."""
        if current_quat is None or origin_quat is None:
            return 0.0
        
        try:
            # Calculate relative rotation quaternion (from origin to current)
            origin_rotation = R.from_quat(origin_quat)
            current_rotation = R.from_quat(current_quat)
            relative_rotation = current_rotation * origin_rotation.inv()
            
            # Project the relative rotation onto the X-axis (pitch)
            # Get the rotation vector (axis-angle representation)
            rotvec = relative_rotation.as_rotvec()
            
            # The X-component of the rotation vector represents rotation around X-axis (pitch)
            x_rotation_rad = rotvec[0]
            x_rotation_deg = np.degrees(x_rotation_rad)
            
            return x_rotation_deg
        except Exception as e:
            logger.warning(f"Error extracting pitch from quaternion: {e}")
            return 0.0 