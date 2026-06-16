# Vendored: reBot official kinematics

These files are copied verbatim from the official reBot control library:

    reBotArm_control_py/reBotArm_control_py/kinematics/
    (https://github.com/  — local source: ~/reBot/reBotArm_control_py)

Files: `robot_model.py`, `forward_kinematics.py`, `inverse_kinematics.py`,
`__init__.py` — Pinocchio-based forward/inverse kinematics for the reBot-DevArm.

They are vendored here so telegrip's reBot teleop is self-contained (no
dependency on the external `~/reBot` checkout). Only runtime dependency is
`pinocchio` (PyPI: `pin`). Do not edit — re-copy from upstream to update.

Used via `telegrip/core/rebot_kinematics.py` (the telegrip adapter).
