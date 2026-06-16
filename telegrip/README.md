# telegrip

The teleoperation application for **VR-teleop**. It supports the SO-ARM100/SO-101
(Feetech) and reBot B601-DM (Damiao CAN) arms via VR or keyboard input.

📖 **Full documentation — installation, configuration, running, and controls — is in
the [repository README](../README.md).**

Quick start (from this directory):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[pybullet,rebot]"      # extras as needed
# set robot.type and ports in config.yaml, then:
telegrip
```

## License

MIT — see [LICENSE](LICENSE). The vendored reBot kinematics retain their original
license; see [telegrip/vendor/rebot_kinematics/NOTICE.md](telegrip/vendor/rebot_kinematics/NOTICE.md).
