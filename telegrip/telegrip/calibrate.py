"""
Standalone calibration entry point.

Run before first use or whenever the arm is remounted / servos are replaced:

    .venv/bin/python -m telegrip.calibrate
    .venv/bin/python -m telegrip.calibrate --arm left
    .venv/bin/python -m telegrip.calibrate --arm right

Or via the installed CLI:

    telegrip-calibrate
    telegrip-calibrate --arm left

The calibration is saved to the paths configured in config.yaml
(robot.left_arm.calibration / robot.right_arm.calibration).
After running, restart the teleoperation system — it will load the
new calibration automatically.
"""

import argparse
import sys
import logging

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SO100 robot arms for accurate teleoperation."
    )
    parser.add_argument(
        "--arm",
        choices=["left", "right", "both"],
        default="both",
        help="Which arm to calibrate (default: both)",
    )
    args = parser.parse_args()

    # Import here so logging is configured first
    from telegrip.config import TelegripConfig
    from telegrip.driver.calibrator import SO100Calibrator

    cfg = TelegripConfig()
    arms_to_run = (
        ["left", "right"] if args.arm == "both"
        else [args.arm]
    )

    print("\n🤖  RoboInn SO100 Arm Calibration")
    print("="*60)
    print("This will guide you through a two-phase calibration:")
    print("  1. Homing   — centre each joint's reported zero")
    print("  2. Range    — record each joint's min / max limits")
    print()
    print("You will need to move the arm by hand. Torque will be")
    print("disabled automatically during calibration.\n")
    print("="*60)

    for arm in arms_to_run:
        if not cfg.arm_enabled.get(arm, True):
            print(f"\n⚠️  {arm.upper()} arm is disabled in config — skipping.")
            continue

        port     = cfg.follower_ports[arm]
        save_path = cfg.calibration_paths.get(arm)

        if save_path is None:
            print(
                f"\n⚠️  No calibration_path configured for {arm} arm in config.yaml.\n"
                f"   Set robot.{arm}_arm.calibration to a file path and re-run."
            )
            continue

        try:
            calibrator = SO100Calibrator(
                port=port,
                arm_id=arm,
                save_path=save_path,
            )
            calibrator.run()
        except ConnectionError as e:
            print(f"\n❌  Cannot connect to {arm} arm on {port}: {e}")
            print("    Check the USB cable and port setting in config.yaml.")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n\n🛑  Calibration interrupted.")
            sys.exit(1)

    print("\n✅  All done. Restart teleoperation to apply new calibration.\n")


if __name__ == "__main__":
    main()
