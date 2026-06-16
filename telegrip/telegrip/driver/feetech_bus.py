"""
Feetech STS3215 bus driver built on top of the official scservo_sdk
(installed as the 'feetech-servo-sdk' package).

All position values exchanged with the public API are raw servo ticks
(integers 0-4095).  Conversion to/from degrees is done in so100.py where
the per-joint calibration data is available.
"""

import logging
from typing import Dict, List, Optional

from scservo_sdk import (
    PortHandler,
    protocol_packet_handler,
    GroupSyncRead,
    GroupSyncWrite,
    COMM_SUCCESS,
    SCS_LOBYTE,
    SCS_HIBYTE,
)

logger = logging.getLogger(__name__)

# STS3215 register map
REG_TORQUE_ENABLE    = 40   # 1 byte
REG_GOAL_POSITION    = 42   # 2 bytes
REG_PRESENT_POSITION = 56   # 2 bytes


class FeetechBus:
    """
    Low-level serial bus driver for Feetech STS3215 servos.

    All position values are raw ticks (0–4095).  The caller is responsible
    for converting between ticks and degrees using per-joint calibration.
    """

    def __init__(self, port: str, baudrate: int = 1_000_000):
        self.port     = port
        self.baudrate = baudrate

        self._port_handler = PortHandler(port)
        self._ph           = protocol_packet_handler()
        self._sync_read:  Optional[GroupSyncRead]  = None
        self._sync_write: Optional[GroupSyncWrite] = None

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #

    def connect(self) -> None:
        self._port_handler.setBaudRate(self.baudrate)
        if not self._port_handler.openPort():
            raise ConnectionError(f"Cannot open serial port {self.port}")
        logger.debug(f"Opened {self.port} at {self.baudrate} baud")

        self._sync_read  = GroupSyncRead(
            self._port_handler, self._ph, REG_PRESENT_POSITION, 2
        )
        self._sync_write = GroupSyncWrite(
            self._port_handler, self._ph, REG_GOAL_POSITION, 2
        )

    def disconnect(self) -> None:
        if self._port_handler.is_open:
            self._port_handler.closePort()
            logger.debug(f"Closed {self.port}")

    @property
    def is_open(self) -> bool:
        return self._port_handler.is_open

    # ------------------------------------------------------------------ #
    # Torque
    # ------------------------------------------------------------------ #

    def enable_torque(self, motor_id: int) -> None:
        self._ph.write1ByteTxOnly(
            self._port_handler, motor_id, REG_TORQUE_ENABLE, 1
        )

    def disable_torque(self, motor_id: int = 0xFE) -> None:
        self._ph.write1ByteTxOnly(
            self._port_handler, motor_id, REG_TORQUE_ENABLE, 0
        )

    # ------------------------------------------------------------------ #
    # Reads  (return raw ticks)
    # ------------------------------------------------------------------ #

    def read_position_ticks(self, motor_id: int) -> Optional[int]:
        """Read present position as raw ticks, or None on failure."""
        data, result, _ = self._ph.read2ByteTxRx(
            self._port_handler, motor_id, REG_PRESENT_POSITION
        )
        if result == COMM_SUCCESS:
            return int(data)
        logger.debug(f"read_position_ticks({motor_id}) failed: result={result}")
        return None

    def sync_read_positions_ticks(self, motor_ids: List[int]) -> Optional[Dict[int, int]]:
        """
        Read present positions for multiple motors in one SYNC_READ.
        Returns {motor_id: raw_ticks} or None on failure.
        """
        sr = self._sync_read
        if sr is None:
            return None

        for mid in motor_ids:
            sr.addParam(mid)

        result = sr.txRxPacket()
        if result != COMM_SUCCESS:
            logger.debug(f"sync_read_positions_ticks failed: result={result}")
            return None

        out = {}
        for mid in motor_ids:
            if sr.isAvailable(mid, REG_PRESENT_POSITION, 2):
                out[mid] = int(sr.getData(mid, REG_PRESENT_POSITION, 2))
            else:
                logger.debug(f"sync_read: no data for motor {mid}")
                return None
        return out

    # ------------------------------------------------------------------ #
    # Writes  (accept raw ticks)
    # ------------------------------------------------------------------ #

    def write_position_ticks(self, motor_id: int, ticks: int) -> None:
        """Send goal position as raw ticks to a single motor."""
        self._ph.write2ByteTxOnly(
            self._port_handler, motor_id, REG_GOAL_POSITION, ticks
        )

    def sync_write_positions_ticks(self, id_ticks: Dict[int, int]) -> None:
        """Send goal positions (raw ticks) to multiple motors via SYNC_WRITE."""
        sw = self._sync_write
        if sw is None:
            return
        sw.clearParam()
        for motor_id, ticks in id_ticks.items():
            sw.addParam(motor_id, [SCS_LOBYTE(ticks), SCS_HIBYTE(ticks)])
        sw.txPacket()
