"""
Phase 1 verification for the official-IK adapter (core/rebot_kinematics.py).

Runs without hardware. Checks that the vendor-backed FK/IK round-trips
accurately and behaves sensibly across seeds / a workspace sweep.

Run:  ../.venv/bin/python tests/test_rebot_kinematics.py   (from telegrip/)
"""

import sys
import numpy as np

from telegrip.core.rebot_kinematics import RebotKinematics


def _angle_err(R_a, R_b):
    """Geodesic rotation error (radians) between two rotation matrices."""
    R = R_a.T @ R_b
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def main() -> int:
    kin = RebotKinematics()
    assert kin.num_joints == 6, f"expected 6-DOF model, got {kin.num_joints}"

    rng = np.random.default_rng(0)
    POS_TOL = 1e-3   # 1 mm
    ROT_TOL = 1e-2   # ~0.57 deg

    # 1) FK determinism
    q = np.array([0.2, -0.5, -0.7, 0.3, 0.4, -0.6])
    p1, r1 = kin.fk(q)
    p2, r2 = kin.fk(q)
    assert np.allclose(p1, p2) and np.allclose(r1, r2), "FK not deterministic"

    # 2) Round-trip over random reachable poses (FK -> IK -> FK)
    n_total, n_ok, n_solved = 50, 0, 0
    worst_pos, worst_rot = 0.0, 0.0
    for _ in range(n_total):
        q_true = rng.uniform(-1.2, 1.2, size=6)
        # keep joints 2,3 in the arm's negative-range to stay reachable
        q_true[1] = rng.uniform(-2.5, -0.2)
        q_true[2] = rng.uniform(-2.5, -0.2)
        pos, rot = kin.fk(q_true)

        q_seed = q_true + rng.uniform(-0.3, 0.3, size=6)
        q_sol, success, err = kin.solve(pos, rot, q_seed)
        n_solved += int(success)

        p_a, r_a = kin.fk(q_sol)
        pe = float(np.linalg.norm(p_a - pos))
        re = _angle_err(r_a, rot)
        worst_pos, worst_rot = max(worst_pos, pe), max(worst_rot, re)
        if pe < POS_TOL and re < ROT_TOL:
            n_ok += 1

    print(f"round-trip: {n_ok}/{n_total} within tol | solver success {n_solved}/{n_total}")
    print(f"worst pos_err={worst_pos*1000:.3f} mm | worst rot_err={np.degrees(worst_rot):.3f} deg")

    # 3) Small Cartesian step from a nominal pose resolves with a small joint move
    q0 = np.array([0.0, -1.0, -1.0, 0.0, 0.0, 0.0])
    p0, r0 = kin.fk(q0)
    q_step, ok_step, _ = kin.solve(p0 + np.array([0.02, 0.0, 0.0]), r0, q0)
    assert ok_step, "IK failed on a 2 cm step"
    p_step, _ = kin.fk(q_step)
    assert np.linalg.norm(p_step - (p0 + np.array([0.02, 0, 0]))) < POS_TOL, "step not reached"

    ok = (n_ok >= int(0.9 * n_total)) and (worst_pos < 5e-3)
    print("PHASE 1 ADAPTER TEST:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
