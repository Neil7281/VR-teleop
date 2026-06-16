"""
Standalone SO100 follower arm driver with lerobot-compatible calibration.

Position conversion formula (matches lerobot FeetechMotorsBus exactly):
    degrees = (raw_ticks - mid) * 360 / 4095
    raw_ticks = int(degrees * 4095 / 360 + mid)
    where mid = (range_min + range_max) / 2  — per-joint, from calibration file

Without calibration lerobot's formula still applies but with mid=2048 which
can be off by 10–35 ° per joint, causing inaccurate IK tracking.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from .feetech_bus import FeetechBus

logger = logging.getLogger(__name__)

_MAX_RES = 4095   # STS3215 resolution – 1 (matches lerobot's max_res)

# Joint name → Feetech motor ID (SO100 wiring)
JOINT_IDS: Dict[str, int] = {
    "shoulder_pan":  1,
    "shoulder_lift": 2,
    "elbow_flex":    3,
    "wrist_flex":    4,
    "wrist_roll":    5,
    "gripper":       6,
}
_ALL_IDS     = list(JOINT_IDS.values())
_ID_TO_JOINT = {v: k for k, v in JOINT_IDS.items()}


# ------------------------------------------------------------------ #
# Per-joint calibration
# ------------------------------------------------------------------ #

@dataclass
class JointCalibration:
    range_min: int
    range_max: int

    @property
    def mid(self) -> float:
        return (self.range_min + self.range_max) / 2.0

    def ticks_to_degrees(self, ticks: int) -> float:
        return (ticks - self.mid) * 360.0 / _MAX_RES

    def degrees_to_ticks(self, degrees: float) -> int:
        raw = int(degrees * _MAX_RES / 360.0 + self.mid)
        return max(self.range_min, min(self.range_max, raw))


def load_calibration(path: str) -> Dict[str, JointCalibration]:
    """Load a lerobot-format calibration JSON and return per-joint objects."""
    with open(path) as f:
        data = json.load(f)
    calib = {}
    for joint, vals in data.items():
        calib[joint] = JointCalibration(
            range_min=int(vals["range_min"]),
            range_max=int(vals["range_max"]),
        )
    logger.info(f"Loaded calibration from {path}")
    return calib


# Fallback: use center=2048 (uncalibrated, may be inaccurate)
_FALLBACK_CALIB = JointCalibration(range_min=1, range_max=4095)


# ------------------------------------------------------------------ #
# Config & driver
# ------------------------------------------------------------------ #

@dataclass
class SOFollowerConfig:
    """Configuration for one SO100 follower arm."""
    port: str
    id: str   = "follower"
    use_degrees: bool = True
    disable_torque_on_disconnect: bool = True
    baudrate: int = 1_000_000
    calibration_path: Optional[str] = None   # path to lerobot-format JSON


class SOFollower:
    """
    High-level driver for one SO100 follower arm.

    Observation dict keys:  "<joint_name>.pos"  (float, degrees)
    Action dict keys:       "<joint_name>.pos"  (float, degrees)

    Position conversion is calibration-aware and matches lerobot exactly.
    Point `calibration_path` at the JSON produced by `lerobot-calibrate`
    (typically ~/.cache/huggingface/lerobot/calibration/robots/so_follower/*.json)
    to get accurate tracking.
    """

    def __init__(self, config: SOFollowerConfig) -> None:
        self.config = config
        self.bus    = FeetechBus(config.port, config.baudrate)
        self._calib: Dict[str, JointCalibration] = {}
        self._last_obs: Optional[Dict[str, float]] = None
        self._connected = False

    # ---------------------------------------------------------------- #
    # Lifecycle
    # ---------------------------------------------------------------- #

    def connect(self) -> None:
        # Load calibration before opening serial (fail fast if file missing)
        if self.config.calibration_path:
            try:
                self._calib = load_calibration(self.config.calibration_path)
            except Exception as exc:
                logger.warning(
                    f"[{self.config.id}] Could not load calibration from "
                    f"{self.config.calibration_path}: {exc}. "
                    f"Falling back to uncalibrated mode (mid=2048)."
                )
        else:
            logger.warning(
                f"[{self.config.id}] No calibration_path set — positions will "
                f"use mid=2048 and may be inaccurate. Set robot.left_arm.calibration "
                f"(or right_arm.calibration) in config.yaml."
            )

        self.bus.connect()
        self._connected = True
        logger.info(f"[{self.config.id}] Connected on {self.config.port}")

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self.config.disable_torque_on_disconnect:
                self.bus.disable_torque()
        except Exception as exc:
            logger.warning(f"[{self.config.id}] Error disabling torque: {exc}")
        finally:
            self.bus.disconnect()
            self._connected = False
            logger.info(f"[{self.config.id}] Disconnected")

    # ---------------------------------------------------------------- #
    # Conversion helpers
    # ---------------------------------------------------------------- #

    def _ticks_to_deg(self, joint: str, ticks: int) -> float:
        return self._calib.get(joint, _FALLBACK_CALIB).ticks_to_degrees(ticks)

    def _deg_to_ticks(self, joint: str, degrees: float) -> int:
        return self._calib.get(joint, _FALLBACK_CALIB).degrees_to_ticks(degrees)

    # ---------------------------------------------------------------- #
    # Read
    # ---------------------------------------------------------------- #

    def get_observation(self) -> Optional[Dict[str, float]]:
        """
        Read all joint positions via SYNC_READ and return calibrated degrees.
        Falls back to last known values on transient failures.
        """
        raw = self.bus.sync_read_positions_ticks(_ALL_IDS)
        if raw is None:
            logger.debug(f"[{self.config.id}] sync_read failed — using cached")
            return self._last_obs

        obs = {
            f"{_ID_TO_JOINT[mid]}.pos": self._ticks_to_deg(_ID_TO_JOINT[mid], ticks)
            for mid, ticks in raw.items()
        }
        self._last_obs = obs
        return obs

    # ---------------------------------------------------------------- #
    # Write
    # ---------------------------------------------------------------- #

    def send_action(self, action: Dict[str, float]) -> None:
        """
        Send calibrated degree values to all joints listed in *action*
        via SYNC_WRITE.
        """
        id_ticks: Dict[int, int] = {}
        for joint, motor_id in JOINT_IDS.items():
            key = f"{joint}.pos"
            if key in action:
                id_ticks[motor_id] = self._deg_to_ticks(joint, float(action[key]))

        if id_ticks:
            self.bus.sync_write_positions_ticks(id_ticks)
