"""
Interactive calibration for SO100 arms — matches lerobot's calibration exactly.

Two-phase process (identical to `lerobot-calibrate` for so_follower):
  Phase 1 – Homing
    • Reset Homing_Offset=0 on all servos
    • User moves arm to the physical centre of its range
    • Compute homing_offset = raw_ticks − 2047 per joint
    • Write Homing_Offset to servo EEPROM
    → After this, every joint reads ≈ 2047 at its physical centre

  Phase 2 – Range of motion
    • User sweeps all joints (except wrist_roll) through their full range
    • Record observed min/max Present_Position ticks
    • wrist_roll gets fixed range 0–4095 (continuous rotation)
    • Write Min/Max_Position_Limit to servo EEPROM
    → mid = (range_min + range_max) / 2, used for degrees conversion

Calibration is saved as a lerobot-compatible JSON at the path in
config.yaml (robot.left_arm.calibration / robot.right_arm.calibration).
"""

import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scservo_sdk import (
    PortHandler,
    protocol_packet_handler,
    COMM_SUCCESS,
    SCS_LOBYTE,
    SCS_HIBYTE,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# STS3215 register addresses
# ------------------------------------------------------------------ #
REG_MIN_POSITION_LIMIT = 9    # 2 bytes, EEPROM
REG_MAX_POSITION_LIMIT = 11   # 2 bytes, EEPROM
REG_HOMING_OFFSET      = 31   # 2 bytes, EEPROM, sign-magnitude (sign bit 11)
REG_TORQUE_ENABLE      = 40   # 1 byte,  SRAM
REG_LOCK               = 55   # 1 byte,  SRAM  (0=unlock EEPROM, 1=lock)
REG_PRESENT_POSITION   = 56   # 2 bytes, SRAM, read-only

_MAX_RES    = 4095   # 4096 − 1
_HALF_TURN  = 2047   # _MAX_RES // 2  → target ticks at arm centre
_SIGN_BIT   = 11     # Homing_Offset sign bit index


# ------------------------------------------------------------------ #
# Sign-magnitude helpers (Feetech convention)
# ------------------------------------------------------------------ #

def _encode_sm(value: int) -> int:
    """Encode a signed integer as sign-magnitude with sign at bit _SIGN_BIT."""
    if value < 0:
        return (-value) | (1 << _SIGN_BIT)
    return value


def _decode_sm(raw: int) -> int:
    """Decode a sign-magnitude value with sign at bit _SIGN_BIT."""
    sign = (raw >> _SIGN_BIT) & 1
    mag  = raw & ((1 << _SIGN_BIT) - 1)
    return -mag if sign else mag


# ------------------------------------------------------------------ #
# Low-level helpers
# ------------------------------------------------------------------ #

def _write1(ph, port, mid: int, reg: int, val: int):
    ph.write1ByteTxOnly(port, mid, reg, val)

def _write2(ph, port, mid: int, reg: int, val: int):
    ph.write2ByteTxOnly(port, mid, reg, val)

def _read2(ph, port, mid: int, reg: int) -> Optional[int]:
    data, result, _ = ph.read2ByteTxRx(port, mid, reg)
    return int(data) if result == COMM_SUCCESS else None

def _unlock_eeprom(ph, port, mid: int):
    _write1(ph, port, mid, REG_TORQUE_ENABLE, 0)
    _write1(ph, port, mid, REG_LOCK, 0)

def _lock_eeprom(ph, port, mid: int):
    _write1(ph, port, mid, REG_LOCK, 1)


# ------------------------------------------------------------------ #
# Joint list (must match so100.py)
# ------------------------------------------------------------------ #

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]
JOINT_IDS = {j: i + 1 for i, j in enumerate(JOINTS)}
FULL_TURN_JOINT = "wrist_roll"   # fixed range 0–4095


# ------------------------------------------------------------------ #
# Calibration routine
# ------------------------------------------------------------------ #

class SO100Calibrator:
    """
    Interactive calibration for one SO100 arm.

    Usage::

        cal = SO100Calibrator("/dev/ttyACM0", arm_id="left",
                              save_path="~/.cache/...arm1.json")
        cal.run()          # interactive — blocks until done
    """

    def __init__(self, port: str, arm_id: str = "arm",
                 save_path: Optional[str] = None,
                 baudrate: int = 1_000_000):
        self.port      = port
        self.arm_id    = arm_id
        self.save_path = Path(save_path).expanduser() if save_path else None
        self.baudrate  = baudrate

        self._ph   = protocol_packet_handler()
        self._port = PortHandler(port)

    # ---------------------------------------------------------------- #
    # Public
    # ---------------------------------------------------------------- #

    def run(self) -> Dict:
        """Run the full calibration and return the calibration dict."""
        self._open()
        try:
            print(f"\n{'='*60}")
            print(f"  Calibrating {self.arm_id.upper()} arm  ({self.port})")
            print(f"{'='*60}\n")

            self._reset_homing_offsets()
            homing_offsets = self._phase1_homing()
            range_mins, range_maxes = self._phase2_range()

            calib = self._build_calib(homing_offsets, range_mins, range_maxes)
            self._write_calibration_to_hardware(calib)
            self._save(calib)

            print(f"\n✅  {self.arm_id.upper()} arm calibration complete.\n")
            return calib
        finally:
            self._close()

    # ---------------------------------------------------------------- #
    # Internal phases
    # ---------------------------------------------------------------- #

    def _phase1_homing(self) -> Dict[str, int]:
        """Ask user to centre the arm, then record homing offsets."""
        print("─── Phase 1: Homing ─────────────────────────────────────")
        print(f"  Torque is OFF on {self.arm_id.upper()} arm.")
        print("  Move ALL joints to the MIDDLE of their range of motion.")
        print("  (A relaxed, roughly centred pose works well.)\n")
        input("  Press ENTER when arm is in position…")

        offsets = {}
        for joint in JOINTS:
            mid_id = JOINT_IDS[joint]
            raw = _read2(self._ph, self._port, mid_id, REG_PRESENT_POSITION)
            if raw is None:
                raise IOError(f"Failed to read position for {joint} (motor {mid_id})")
            offsets[joint] = raw - _HALF_TURN
            print(f"  {joint:<16} raw={raw:4d}  homing_offset={offsets[joint]:+d}")

        print()
        return offsets

    def _phase2_range(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Ask user to sweep every joint; record observed min/max."""
        print("─── Phase 2: Range of motion ────────────────────────────")
        print(f"  Torque is still OFF on {self.arm_id.upper()} arm.")
        print(f"  Slowly move ALL joints (except '{FULL_TURN_JOINT}') through")
        print("  their FULL range of motion — from hard stop to hard stop.")
        print(f"  '{FULL_TURN_JOINT}' will get a fixed range 0–{_MAX_RES}.")
        print("\n  Live positions are shown below. Press ENTER when done.\n")

        mins:  Dict[str, int] = {j: _MAX_RES for j in JOINTS if j != FULL_TURN_JOINT}
        maxes: Dict[str, int] = {j: 0        for j in JOINTS if j != FULL_TURN_JOINT}

        # Poll until Enter is pressed (non-blocking check via select)
        import select
        import os

        # Make stdin non-blocking in a portable way
        done = False

        def _check_enter():
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                return True
            return False

        print(f"  {'JOINT':<16} {'POS':>5}  {'MIN':>5}  {'MAX':>5}")
        print(f"  {'-'*16} {'-----':>5}  {'-----':>5}  {'-----':>5}")

        while not done:
            row = f"\r  "
            for joint in JOINTS:
                if joint == FULL_TURN_JOINT:
                    continue
                mid_id = JOINT_IDS[joint]
                raw = _read2(self._ph, self._port, mid_id, REG_PRESENT_POSITION)
                if raw is not None:
                    if raw < mins[joint]:  mins[joint]  = raw
                    if raw > maxes[joint]: maxes[joint] = raw
            # Reprint current state
            sys.stdout.write("\033[A" * (len(JOINTS)))  # move cursor up
            for joint in JOINTS:
                if joint == FULL_TURN_JOINT:
                    raw_str = f"  {joint:<16} {'--':>5}  {'0':>5}  {str(_MAX_RES):>5}"
                else:
                    mid_id = JOINT_IDS[joint]
                    raw = _read2(self._ph, self._port, mid_id, REG_PRESENT_POSITION)
                    cur = raw if raw is not None else 0
                    raw_str = (
                        f"  {joint:<16} {cur:5d}  "
                        f"{mins[joint]:5d}  {maxes[joint]:5d}"
                    )
                print(raw_str)

            time.sleep(0.08)
            done = _check_enter()

        # wrist_roll gets full range
        mins[FULL_TURN_JOINT]  = 0
        maxes[FULL_TURN_JOINT] = _MAX_RES

        print("\n  Recorded ranges:")
        for joint in JOINTS:
            print(f"  {joint:<16} min={mins[joint]:4d}  max={maxes[joint]:4d}")
        print()
        return mins, maxes

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #

    def _reset_homing_offsets(self):
        """Write Homing_Offset=0 to all motors (so we read raw ADC ticks)."""
        for joint in JOINTS:
            mid_id = JOINT_IDS[joint]
            _unlock_eeprom(self._ph, self._port, mid_id)
            _write2(self._ph, self._port, mid_id, REG_HOMING_OFFSET, 0)
        time.sleep(0.1)  # let EEPROM settle

    def _write_calibration_to_hardware(self, calib: Dict):
        """Write homing offsets and position limits to servo EEPROM."""
        print("  Writing calibration to servo EEPROM…")
        for joint, data in calib.items():
            mid_id = JOINT_IDS[joint]
            _unlock_eeprom(self._ph, self._port, mid_id)
            _write2(self._ph, self._port, mid_id,
                    REG_HOMING_OFFSET, _encode_sm(data["homing_offset"]))
            _write2(self._ph, self._port, mid_id,
                    REG_MIN_POSITION_LIMIT, data["range_min"])
            _write2(self._ph, self._port, mid_id,
                    REG_MAX_POSITION_LIMIT, data["range_max"])
            _lock_eeprom(self._ph, self._port, mid_id)
        time.sleep(0.15)

    def _build_calib(self, homing_offsets, range_mins, range_maxes) -> Dict:
        calib = {}
        for joint in JOINTS:
            calib[joint] = {
                "id":            JOINT_IDS[joint],
                "drive_mode":    0,
                "homing_offset": homing_offsets[joint],
                "range_min":     range_mins[joint],
                "range_max":     range_maxes[joint],
            }
        return calib

    def _save(self, calib: Dict):
        if self.save_path is None:
            print("  (No save_path set — calibration NOT saved to disk.)")
            return
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(calib, f, indent=4)
        print(f"  Calibration saved → {self.save_path}")

    def _open(self):
        self._port.setBaudRate(self.baudrate)
        if not self._port.openPort():
            raise ConnectionError(f"Cannot open {self.port}")
        # Disable torque so arm can be moved by hand
        for joint in JOINTS:
            _write1(self._ph, self._port, JOINT_IDS[joint], REG_TORQUE_ENABLE, 0)

    def _close(self):
        if self._port.is_open:
            self._port.closePort()
