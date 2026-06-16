"""
Phase 2/3 integration test — reBot pose-based control pipeline, no hardware/VR.

Exercises:
  - RobotInterface.get_current_end_effector_pose / update_arm_from_pose (Phase 2)
  - control_loop._vr_delta_quat_to_robot_rotation (Phase 3 mapping)
  - a simulated engage -> position+orientation delta -> IK -> FK round-trip

Run:  ../.venv/bin/python tests/test_rebot_pose_control.py   (from telegrip/)
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation as R

from telegrip.config import TelegripConfig
from telegrip.core.robot_interface import RobotInterface
from telegrip.control_loop import _vr_delta_quat_to_robot_rotation, _VR_TO_ROBOT_BASIS


def _angle_err(Ra, Rb):
    Rd = Ra.T @ Rb
    return float(np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0)))


def main() -> int:
    cfg = TelegripConfig()
    assert cfg.is_rebot, "config.yaml robot.type must be 'rebot' for this test"
    cfg.enable_robot = False  # no hardware
    # Disable the per-joint speed cap so the IK round-trip checks below reach the
    # target in a single call (the cap is exercised in its own unit test).
    cfg.rebot_joint_max_vel_deg_s = ()

    ri = RobotInterface(cfg)

    # --- Phase 3: rotation-mapping sanity -------------------------------------
    assert np.allclose(_VR_TO_ROBOT_BASIS @ _VR_TO_ROBOT_BASIS.T, np.eye(3)), "M not orthogonal"
    assert abs(np.linalg.det(_VR_TO_ROBOT_BASIS) - 1.0) < 1e-9, "M not a proper rotation"
    # identity controller delta -> identity robot rotation
    Rid = _vr_delta_quat_to_robot_rotation(np.array([0, 0, 0, 1.0]))
    assert np.allclose(Rid, np.eye(3), atol=1e-9), "identity delta must map to identity"
    # a 30deg VR yaw maps to a proper rotation in robot frame
    Rmap = _vr_delta_quat_to_robot_rotation(R.from_euler('y', 30, degrees=True).as_quat())
    assert abs(np.linalg.det(Rmap) - 1.0) < 1e-9 and np.allclose(Rmap @ Rmap.T, np.eye(3))
    print("Phase 3 mapping checks: OK")

    # --- Phase 2: FK/IK pose round-trip through RobotInterface ----------------
    # Put the arm in a non-singular start config (degrees), gripper at 0.
    start = np.array([10.0, -60.0, -70.0, 15.0, 20.0, -25.0, 0.0])
    ri.right_arm_angles = start.copy()

    pos0, rot0 = ri.get_current_end_effector_pose("right")

    # Simulated engage origin + a VR delta: move +3cm x / -2cm z, rotate 15deg.
    target_pos = pos0 + np.array([0.03, 0.0, -0.02])
    delta_rot = R.from_euler('xyz', [10, -8, 12], degrees=True).as_matrix()
    target_rot = delta_rot @ rot0

    ok = ri.update_arm_from_pose("right", target_pos, target_rot)
    # gripper (index 6) must be preserved
    assert ri.right_arm_angles[6] == start[6], "gripper must be untouched by pose IK"

    pos1, rot1 = ri.get_current_end_effector_pose("right")
    pos_err = float(np.linalg.norm(pos1 - target_pos))
    rot_err = _angle_err(rot1, target_rot)
    print(f"pose round-trip: success={ok} pos_err={pos_err*1000:.3f} mm rot_err={np.degrees(rot_err):.3f} deg")

    # --- Multi-step relative teleop simulation (origin held, deltas applied) ---
    origin_rot = rot0.copy()
    worst = 0.0
    for i in range(1, 6):
        tpos = pos0 + np.array([0.01 * i, 0.005 * i, -0.004 * i])
        # cumulative controller rotation in VR frame -> robot frame -> applied to origin
        dq = R.from_euler('y', 5 * i, degrees=True).as_quat()
        trot = _vr_delta_quat_to_robot_rotation(dq) @ origin_rot
        ri.update_arm_from_pose("right", tpos, trot)
        p, r = ri.get_current_end_effector_pose("right")
        worst = max(worst, float(np.linalg.norm(p - tpos)))
    print(f"multi-step worst pos_err: {worst*1000:.3f} mm")

    # --- Path following: trace an L-shaped path via the waypoint queue ----------
    ri.right_arm_angles = start.copy()
    ri.anchor_rebot_tracking("right")
    p0, r0 = ri.get_current_end_effector_pose("right")
    corner = p0 + np.array([0.10, 0.0, 0.0])          # go +x 10 cm ...
    end = corner + np.array([0.0, 0.08, 0.0])         # ... then +y 8 cm (L shape)
    max_lin, max_ang = 0.5 * 0.01, np.radians(180) * 0.01  # per 10 ms cycle
    min_lin, min_ang, max_wp = 0.005, np.radians(2.0), 60

    # Feed the two path segments densely (as the hand would stream them).
    def seg(a, b, n):
        return [a + (b - a) * (k / n) for k in range(1, n + 1)]
    hand_path = seg(p0, corner, 40) + seg(corner, end, 40)

    prev = p0.copy()
    prev_q = ri.right_arm_angles[:6].copy()
    prev_qv = np.zeros(6)
    max_step = 0.0
    min_dist_to_corner = 1e9
    peak_acc = 0.0
    ee_trace = []
    fed = 0
    for i in range(3000):
        if fed < len(hand_path):           # stream a couple of hand samples per cycle
            for _ in range(2):
                if fed < len(hand_path):
                    ri.add_waypoint("right", hand_path[fed], r0, min_lin, min_ang, max_wp)
                    fed += 1
        ri.track_path("right", max_lin, max_ang)
        p, _ = ri.get_current_end_effector_pose("right")
        max_step = max(max_step, float(np.linalg.norm(p - prev)))
        min_dist_to_corner = min(min_dist_to_corner, float(np.linalg.norm(p - corner)))
        prev = p.copy()
        # joint-space acceleration (the OTG must keep this bounded → smooth)
        q = ri.right_arm_angles[:6].copy()
        qv = (q - prev_q) / 0.01
        peak_acc = max(peak_acc, float(np.max(np.abs((qv - prev_qv) / 0.01))))
        prev_q, prev_qv = q, qv
        ee_trace.append(p.copy())
        if fed >= len(hand_path) and np.linalg.norm(p - end) < 2e-3:
            break

    # With jerk-limited OTG the corner is intentionally ROUNDED (not cut to the
    # endpoint): it follows the path but smooths sharp turns, and joint accel
    # stays bounded by the OTG cap.
    step_ok = max_step <= max_lin * 2.0
    follows_ok = min_dist_to_corner < 0.05      # tracks the path through the corner region
    reached_ok = np.linalg.norm(ee_trace[-1] - end) < 2e-3
    acc_ok = peak_acc <= cfg.rebot_otg_max_accel_deg_s2 * 1.6
    print(f"path+OTG: max EE step={max_step*1000:.2f} mm | corner rounding={min_dist_to_corner*1000:.1f} mm | "
          f"peak joint acc={peak_acc:.0f} deg/s^2 (cap {cfg.rebot_otg_max_accel_deg_s2:.0f}) | reached={reached_ok}")

    # --- MIT impedance command path (gravity FF + OTG velocity) ----------------
    cfg.rebot_control_mode = "mit"
    ri.right_arm_angles = np.array([10., -60, -70, 15, 20, -25, 0.])
    ri.anchor_rebot_tracking("right")
    pm, rm = ri.get_current_end_effector_pose("right")
    for _ in range(8):
        ri.add_waypoint("right", pm + np.array([0.04, 0, 0]), rm, min_lin, min_ang, max_wp)
        ri.track_path("right", max_lin, max_ang)

    class _MockDriver:
        def send_action_mit(self, pos, vel, tau):
            self.pos, self.vel, self.tau = pos, vel, tau

    mock = _MockDriver()
    ri._send_to_driver(mock, "right")
    mit_ok = (
        len(mock.tau) == 6 and len(mock.vel) == 6 and "gripper" in mock.pos
        and abs(mock.tau["elbow_flex"]) > 1.0          # gravity FF non-trivial on elbow
        and abs(mock.tau["shoulder_pan"]) < 0.5        # ~0 about vertical axis
    )
    print(f"MIT: gravity elbow={mock.tau['elbow_flex']:.2f}Nm pan={mock.tau['shoulder_pan']:.2f}Nm | "
          f"vel/tau dims ok={len(mock.vel)==6 and len(mock.tau)==6}")

    ok_all = (pos_err < 1e-3) and (rot_err < 1e-2) and (worst < 1e-3) and step_ok and follows_ok and reached_ok and acc_ok and mit_ok
    print("PHASE 2/3 INTEGRATION TEST:", "PASS ✅" if ok_all else "FAIL ❌")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
