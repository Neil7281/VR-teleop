"""
reBot B601-DM follower arm driver.

Uses Seeed's motorbridge SDK (pip install motorbridge) — the same library
used by lerobot-robot-seeed-b601.  Two CAN adapter modes are supported:

  can_adapter="damiao"     → MotorBridgeController.from_dm_serial()
                             Damiao USB-to-serial bridge on /dev/ttyACM0,
                             serial baud 921600 (NOT standard CAN-over-serial).

  can_adapter="socketcan"  → MotorBridgeController(channel="can0")
                             Hardware SocketCAN interface.

Joints 0-5 run in POS_VEL mode; gripper runs in FORCE_POS mode.
motorbridge works internally in radians; this driver converts to/from degrees
to match the rest of telegrip.

Official motor model and CAN-ID reference:
  github.com/Seeed-Projects/lerobot-robot-seeed-b601
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# motorbridge import (pip install motorbridge)
# ---------------------------------------------------------------------------
try:
    from motorbridge import Controller as _MBController, Mode as _MBMode
    _MB_AVAILABLE = True
except ImportError:
    _MB_AVAILABLE = False
    logger.warning(
        "motorbridge not available — reBot driver will not function. "
        "Install it with:  pip install motorbridge"
    )

# ---------------------------------------------------------------------------
# Official motor table  (send_can_id, recv_can_id, motorbridge_model_str)
# From seeed_b601_dm_follower.py in lerobot-robot-seeed-b601
# ---------------------------------------------------------------------------
_DEFAULT_MOTOR_TABLE: Dict[str, Tuple[int, int, str]] = {
    "shoulder_pan":  (0x01, 0x11, "4340P"),   # DM4340P — high-torque shoulder
    "shoulder_lift": (0x02, 0x12, "4340P"),
    "elbow_flex":    (0x03, 0x13, "4340P"),
    "wrist_flex":    (0x04, 0x14, "4310"),    # DM4310 — wrist joints
    "wrist_yaw":     (0x05, 0x15, "4310"),
    "wrist_roll":    (0x06, 0x16, "4310"),
    "gripper":       (0x07, 0x17, "4310"),
}

# Official soft joint limits (degrees) from config_seeed_b601_dm_follower.py
_DEFAULT_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "shoulder_pan":  (-145.0, 145.0),
    "shoulder_lift": (-170.0,   1.0),
    "elbow_flex":    (-200.0,   1.0),
    "wrist_flex":    ( -80.0,  90.0),
    "wrist_yaw":     ( -90.0,  90.0),
    "wrist_roll":    ( -90.0,  90.0),
    "gripper":       (-270.0,   0.0),
}

# Joint order — must match REBOT_JOINT_NAMES in config.py
JOINT_NAMES: List[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
]

# Target velocity for POS_VEL mode (degrees/s); converted to rad/s at send time.
# At 100 Hz control with ~3°/step typical movement, 250°/s lets the motor
# track the target without lagging behind.
_DEFAULT_VEL_DEG_S: List[float] = [250.0] * 7

# Gripper force ratio for FORCE_POS mode
_GRIPPER_FORCE_RATIO: float = 0.1


@dataclass
class RebotFollowerConfig:
    """Configuration for a single reBot B601-DM follower arm."""

    port: str
    # "damiao"    → Damiao USB-to-serial bridge (default for B601-DM)
    # "socketcan" → Hardware SocketCAN (can0, can1, …)
    can_adapter: str = "damiao"
    # Serial baud rate — only used when can_adapter="damiao"
    dm_serial_baud: int = 921_600

    id: str = "rebot_follower"
    disable_torque_on_disconnect: bool = True

    motor_table: Dict[str, Tuple[int, int, str]] = field(
        default_factory=lambda: dict(_DEFAULT_MOTOR_TABLE)
    )
    joint_limits: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_JOINT_LIMITS)
    )
    vel_deg_s: List[float] = field(
        default_factory=lambda: list(_DEFAULT_VEL_DEG_S)
    )
    gripper_force_ratio: float = _GRIPPER_FORCE_RATIO

    # Joint control mode: "pos_vel" (position + velocity-limit) or "mit"
    # (impedance: pos+vel reference, per-joint kp/kd, feedforward torque).
    # The gripper always runs FORCE_POS regardless.
    control_mode: str = "pos_vel"
    # Per-joint MIT gains (manufacturer defaults: DM4340 base, DM4310 wrist).
    mit_kp: List[float] = field(default_factory=lambda: [120, 120, 120, 18, 18, 18])
    mit_kd: List[float] = field(default_factory=lambda: [8, 8, 8, 2, 2, 2])


class RebotFollower:
    """
    High-level driver for one reBot B601-DM follower arm.

    Observation dict keys:  "<joint_name>.pos"  (float, degrees)
    Action dict keys:       "<joint_name>.pos"  (float, degrees)
    """

    # Maximum send rate — motorbridge/Damiao can handle ~100 Hz but
    # 125 Hz is a safe upper bound.
    _MIN_SEND_INTERVAL: float = 0.008

    def __init__(self, config: RebotFollowerConfig) -> None:
        if not _MB_AVAILABLE:
            raise ImportError(
                "motorbridge is not installed. Run:  pip install motorbridge"
            )
        self.config = config
        self._bus = None
        self._motors: Dict[str, object] = {}
        self._connected = False
        self._last_send_time: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        """Open CAN/serial connection, add motors, set control modes."""
        logger.info(
            f"[{self.config.id}] Connecting via {self.config.can_adapter} "
            f"on {self.config.port}"
        )

        if self.config.can_adapter == "damiao":
            self._bus = _MBController.from_dm_serial(
                serial_port=self.config.port,
                baud=self.config.dm_serial_baud,
            )
        elif self.config.can_adapter == "socketcan":
            self._bus = _MBController(channel=self.config.port)
        else:
            raise ValueError(
                f"Unknown can_adapter '{self.config.can_adapter}'. "
                "Use 'damiao' or 'socketcan'."
            )

        # Register motors with the bus
        for joint, (send_id, recv_id, model_str) in self.config.motor_table.items():
            self._motors[joint] = self._bus.add_damiao_motor(send_id, recv_id, model_str)

        # Enable and configure
        self._bus.enable_all()
        self._configure_modes()

        self._connected = True
        logger.info(f"[{self.config.id}] Connected — {len(self._motors)} motors enabled")

    def _configure_modes(self) -> None:
        """Set joint mode (POS_VEL or MIT per config), FORCE_POS for gripper."""
        joint_mode = _MBMode.MIT if self.config.control_mode == "mit" else _MBMode.POS_VEL
        for joint, motor in self._motors.items():
            mode = _MBMode.FORCE_POS if joint == "gripper" else joint_mode
            for attempt in range(10):
                try:
                    motor.ensure_mode(mode)
                    break
                except Exception:
                    if attempt == 9:
                        logger.warning(
                            f"[{self.config.id}] Could not set mode for {joint}"
                        )
                    time.sleep(0.01)

    def calibrate(self, prompt: bool = True) -> None:
        """
        Zero-pose calibration for the Damiao motors.

        Mirrors the official ``SeeedB601DMFollower.calibrate()`` flow: torque is
        disabled so the arm can be moved by hand, the operator places it at the
        defined ZERO POSE, then each motor's *current* physical position is set
        as its 0° reference via ``set_zero_position()``.

        This corrects each joint's zero OFFSET. It does NOT reverse a motor's
        rotation direction — if a joint still moves mirrored after zeroing, that
        is a URDF axis-vs-motor-direction mismatch, fixed in the URDF, not here.

        The zero is set in motor RAM only (no ``store_parameters``), so it must
        be re-run after each power cycle — matching the official driver.

        Requires the bus to be connected (``connect()`` first).
        """
        if not self._connected or self._bus is None:
            raise RuntimeError(
                f"[{self.config.id}] calibrate() requires an open connection; "
                "call connect() first."
            )

        # Torque off so the operator can move the arm freely by hand.
        self._bus.disable_all()
        logger.info(f"[{self.config.id}] Torque disabled for calibration.")

        if prompt:
            print(
                f"\n=== reBot zero-pose calibration  [{self.config.id}] ===\n"
                "Manually move the arm to its ZERO POSE (the default sit-down\n"
                "position from the B601 manual) and CLOSE the gripper.\n"
            )
            input("Press ENTER when the arm is at the zero pose... ")

        for joint, motor in self._motors.items():
            try:
                motor.set_zero_position()
                time.sleep(0.1)
                logger.info(f"[{self.config.id}] {joint}: zero position set.")
            except Exception as exc:
                logger.error(
                    f"[{self.config.id}] Failed to zero {joint}: {exc}"
                )

        # Restore torque + control modes so teleop can proceed immediately.
        self._bus.enable_all()
        self._configure_modes()
        logger.info(f"[{self.config.id}] Calibration complete — motors re-enabled.")

    def disconnect(self) -> None:
        """Disable motors and close the connection."""
        if not self._connected or self._bus is None:
            return
        try:
            for motor in self._motors.values():
                if self.config.disable_torque_on_disconnect:
                    try:
                        motor.disable()
                    except Exception:
                        pass
                try:
                    motor.clear_error()
                    motor.close()
                except Exception:
                    pass
            self._bus.close()
        except Exception as exc:
            logger.warning(f"[{self.config.id}] Error during disconnect: {exc}")
        finally:
            self._bus = None
            self._motors = {}
            self._connected = False
            logger.info(f"[{self.config.id}] Disconnected")

    # ------------------------------------------------------------------ #
    # State read / write                                                   #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> Optional[Dict[str, float]]:
        """
        Read all joint positions.
        Returns ``{"<joint>.pos": <degrees>, ...}`` or ``None`` on error.
        """
        if not self._connected or self._bus is None:
            return None
        try:
            for motor in self._motors.values():
                motor.request_feedback()
            try:
                self._bus.poll_feedback_once()
            except Exception:
                pass

            obs: Dict[str, float] = {}
            for joint in JOINT_NAMES:
                state = self._motors[joint].get_state()
                obs[f"{joint}.pos"] = math.degrees(state.pos) if state else 0.0
            return obs
        except Exception as exc:
            logger.debug(f"[{self.config.id}] get_observation failed: {exc}")
            return None

    def send_action(self, action: Dict[str, float]) -> None:
        """
        Send goal positions (degrees) to joints listed in *action*.
        Keys must be ``"<joint_name>.pos"``.
        """
        if not self._connected or self._bus is None:
            return

        now = time.monotonic()
        if now - self._last_send_time < self._MIN_SEND_INTERVAL:
            return

        try:
            for idx, joint in enumerate(JOINT_NAMES):
                key = f"{joint}.pos"
                if key not in action:
                    continue

                pos_deg = float(action[key])

                # Clamp to official soft limits
                lo, hi = self.config.joint_limits.get(joint, (-360.0, 360.0))
                pos_deg = max(lo, min(hi, pos_deg))

                pos_rad = math.radians(pos_deg)
                vel_rad = math.radians(
                    self.config.vel_deg_s[idx]
                    if idx < len(self.config.vel_deg_s)
                    else 150.0
                )

                motor = self._motors.get(joint)
                if motor is None:
                    continue

                if joint == "gripper":
                    motor.send_force_pos(
                        pos_rad, vel_rad, self.config.gripper_force_ratio
                    )
                else:
                    motor.send_pos_vel(pos_rad, vel_rad)

            self._last_send_time = now
        except Exception as exc:
            logger.debug(f"[{self.config.id}] send_action failed: {exc}")

    def send_action_mit(self, positions_deg: Dict[str, float],
                        velocities_deg_s: Dict[str, float],
                        torques_nm: Dict[str, float]) -> None:
        """
        MIT-mode impedance command for joints 1-6 (gripper stays FORCE_POS).

        Per joint the motor applies: tau = kp*(pos_ref - q) + kd*(vel_ref - q̇)
        + tau_ff, with kp/kd from config and tau_ff the gravity feedforward.
        Keys are joint names (no ``.pos`` suffix).
        """
        if not self._connected or self._bus is None:
            return
        now = time.monotonic()
        if now - self._last_send_time < self._MIN_SEND_INTERVAL:
            return
        try:
            for idx, joint in enumerate(JOINT_NAMES):
                motor = self._motors.get(joint)
                if motor is None:
                    continue

                if joint == "gripper":
                    if joint in positions_deg:
                        lo, hi = self.config.joint_limits.get(joint, (-360.0, 360.0))
                        pos_deg = max(lo, min(hi, float(positions_deg[joint])))
                        vel = math.radians(
                            self.config.vel_deg_s[idx]
                            if idx < len(self.config.vel_deg_s) else 150.0
                        )
                        motor.send_force_pos(
                            math.radians(pos_deg), vel, self.config.gripper_force_ratio
                        )
                    continue

                if joint not in positions_deg:
                    continue
                lo, hi = self.config.joint_limits.get(joint, (-360.0, 360.0))
                pos_deg = max(lo, min(hi, float(positions_deg[joint])))
                pos_rad = math.radians(pos_deg)
                vel_rad = math.radians(float(velocities_deg_s.get(joint, 0.0)))
                kp = float(self.config.mit_kp[idx]) if idx < len(self.config.mit_kp) else 0.0
                kd = float(self.config.mit_kd[idx]) if idx < len(self.config.mit_kd) else 1.0
                tau = float(torques_nm.get(joint, 0.0))
                motor.send_mit(pos_rad, vel_rad, kp, kd, tau)

            self._last_send_time = now
        except Exception as exc:
            logger.debug(f"[{self.config.id}] send_action_mit failed: {exc}")

    @property
    def is_connected(self) -> bool:
        return self._connected
