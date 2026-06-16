#!/usr/bin/env python3
"""
Zero-pose calibration for the reBot B601-DM follower arm(s).

Run this ONCE in a terminal before starting teleop (the same way SO-101 is
calibrated separately). It connects to each enabled reBot arm, asks you to
place it at the defined ZERO POSE, then sets that physical position as 0° for
every Damiao motor via ``set_zero_position()``.

This fixes each joint's zero OFFSET so the arm's reported angles line up with
the URDF/IK model. It does NOT reverse a motor's rotation direction — if a
joint still moves mirrored after calibration, that is a URDF axis mismatch and
must be fixed in the URDF, not here.

The zero lives in motor RAM only (no flash write), so re-run this after any
power cycle. Do not power-cycle the arm between calibrating and starting teleop.

Usage:
    python calibrate_rebot.py            # calibrate all enabled arms
    python calibrate_rebot.py --arm left
    python calibrate_rebot.py --arm right
"""

import argparse
import logging
import sys

from telegrip.config import TelegripConfig
from telegrip.core.robot_interface import RobotInterface
from telegrip.driver.rebot import RebotFollower

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("calibrate_rebot")


def _calibrate_arm(arm: str, cfg) -> bool:
    """Connect, run zero-pose calibration, and disconnect one arm."""
    if cfg is None:
        logger.info(f"{arm} arm disabled in config.yaml — skipping.")
        return True

    logger.info(f"Calibrating {arm} arm on {cfg.port} (id={cfg.id})...")
    robot = RebotFollower(cfg)
    try:
        robot.connect()
        robot.calibrate(prompt=True)
        return True
    except Exception as exc:
        logger.error(f"{arm} arm calibration failed: {exc}")
        return False
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm(s) to calibrate (default: both enabled arms).",
    )
    args = parser.parse_args()

    config = TelegripConfig()
    if not config.is_rebot:
        logger.error(
            f"robot.type is '{config.robot_type}', not 'rebot'. "
            "Set robot.type: rebot in config.yaml before calibrating."
        )
        return 1

    # Reuse the exact same per-arm config plumbing that teleop uses.
    interface = RobotInterface(config)
    left_config, right_config = interface.setup_robot_configs()

    arms = {"left": left_config, "right": right_config}
    if args.arm != "both":
        arms = {args.arm: arms[args.arm]}

    print(
        "\nreBot zero-pose calibration\n"
        "Each motor's CURRENT position becomes 0°. Re-run after any power cycle\n"
        "and do not power-cycle the arm before starting teleop.\n"
    )

    ok = True
    for arm, cfg in arms.items():
        ok = _calibrate_arm(arm, cfg) and ok

    if ok:
        logger.info("✅ Calibration finished. You can now start teleop.")
        return 0
    logger.error("❌ Calibration finished with errors — see messages above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
