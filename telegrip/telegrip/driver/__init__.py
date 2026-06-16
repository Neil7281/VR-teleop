"""
Standalone driver for Feetech STS3215 servo motors and SO100 robot arm.
No external robotics framework dependency required.
"""

from .so100 import SOFollower, SOFollowerConfig
from .feetech_bus import FeetechBus

__all__ = ["SOFollower", "SOFollowerConfig", "FeetechBus"]
