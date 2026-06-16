"""
reBot B601-DM joint-direction diagnostic.

Commands small +/- steps on shoulder_pan, shoulder_lift, and elbow_flex
(one at a time, returning to the starting position after each step) and
reports the angle reported back by the motor, so you can see exactly
which joints move opposite to the commanded direction and by how much.

Run with the arm powered, torque-enabled, and clear of obstacles:

    .venv/bin/python -m telegrip.diagnose_rebot --arm right
    .venv/bin/python -m telegrip.diagnose_rebot --arm left --delta 6 --vel 15

Press Ctrl+C at any time — the arm is returned to its starting position
and torque is disabled before exit.
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.WARNING, format="%(message)s")

JOINTS_TO_TEST = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
SETTLE_TIME_S = 1.5


def main():
    parser = argparse.ArgumentParser(description="Diagnose reBot joint-direction mapping.")
    parser.add_argument("--arm", choices=["left", "right"], default="right")
    parser.add_argument("--delta", type=float, default=8.0, help="Test step size in degrees")
    parser.add_argument("--vel", type=float, default=15.0, help="Joint velocity for the test, deg/s")
    args = parser.parse_args()

    from .config import TelegripConfig, _config_data
    from .driver.rebot import RebotFollower, RebotFollowerConfig, JOINT_NAMES

    cfg = TelegripConfig()
    if not cfg.is_rebot:
        print("robot.type is not 'rebot' in config.yaml — aborting.")
        return

    arm_key = "left_arm" if args.arm == "left" else "right_arm"
    arm_cfg = _config_data["robot"].get(arm_key, {})

    follower_cfg = RebotFollowerConfig(
        port=cfg.follower_ports[args.arm],
        id=f"{args.arm}_diagnostic",
        can_adapter=arm_cfg.get("can_adapter", "damiao"),
        dm_serial_baud=arm_cfg.get("dm_serial_baud", 921_600),
        vel_deg_s=[args.vel] * 7,
    )

    print(f"\nConnecting to {args.arm} arm on {follower_cfg.port} ({follower_cfg.can_adapter})...")
    robot = RebotFollower(follower_cfg)
    robot.connect()
    print("Connected.\n")

    try:
        baseline = robot.get_observation()
        if baseline is None:
            print("Could not read initial observation — aborting.")
            return

        print("Baseline position (deg):")
        for j in JOINT_NAMES:
            print(f"  {j:14s} {baseline[f'{j}.pos']:7.2f}")
        print()

        for joint in JOINTS_TO_TEST:
            base_angle = baseline[f"{joint}.pos"]
            print(f"=== {joint} ===  (baseline {base_angle:.2f} deg)")
            input(f"Press ENTER to test {joint} (+/-{args.delta} deg)...")

            for sign, label in [(+1, "+"), (-1, "-")]:
                target = base_angle + sign * args.delta
                action = dict(baseline)
                action[f"{joint}.pos"] = target
                robot.send_action({k: v for k, v in action.items()})
                time.sleep(SETTLE_TIME_S)

                obs = robot.get_observation()
                actual = obs[f"{joint}.pos"]
                measured_delta = actual - base_angle
                commanded_delta = target - base_angle

                if abs(measured_delta) < 0.5:
                    verdict = "NO MOVEMENT"
                elif (measured_delta > 0) == (commanded_delta > 0):
                    verdict = "matches commanded direction"
                else:
                    verdict = "*** OPPOSITE of commanded direction ***"

                print(
                    f"  commanded {label}{args.delta:.1f} deg -> "
                    f"target {target:7.2f}, actual {actual:7.2f}, "
                    f"measured delta {measured_delta:+6.2f}  ({verdict})"
                )

                # Return to baseline before the next step
                action[f"{joint}.pos"] = base_angle
                robot.send_action({k: v for k, v in action.items()})
                time.sleep(SETTLE_TIME_S)

            print()

        print("Diagnostic complete. Returning to baseline and disconnecting.")

    except KeyboardInterrupt:
        print("\nInterrupted — returning to baseline.")
        try:
            robot.send_action(baseline)
            time.sleep(SETTLE_TIME_S)
        except Exception:
            pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
