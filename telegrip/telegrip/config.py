"""
Configuration module for the unified teleoperation system.
Loads configuration from config.yaml file with fallback to default values.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import logging
from .utils import get_absolute_path, get_project_root

logger = logging.getLogger(__name__)

# Default configuration values (fallback if YAML file doesn't exist)
DEFAULT_CONFIG = {
    "network": {
        "https_port": 8443,
        "websocket_port": 8442,
        "host_ip": "0.0.0.0"
    },
    "ssl": {
        "certfile": "cert.pem",
        "keyfile": "key.pem"
    },
    "robot": {
        "type": "so100",       # "so100" or "rebot"
        "left_arm": {
            "name": "Left Arm",
            "port": "/dev/ttyACM0",
            "enabled": True,
            "calibration": None,
            # reBot-only fields
            "can_adapter": "damiao",   # "damiao" (USB serial bridge) or "socketcan"
            "dm_serial_baud": 921600,  # serial baud for Damiao bridge
        },
        "right_arm": {
            "name": "Right Arm",
            "port": "/dev/ttyACM1",
            "enabled": True,
            "calibration": None,
            "can_adapter": "damiao",
            "dm_serial_baud": 921600,
        },
        "vr_to_robot_scale": 1.0,
        "send_interval": 0.05,
        # reBot motion tuning (live-tunable in config.yaml)
        "rebot_motor_velocity_deg_s": 250.0,  # POS_VEL speed cap per joint
        "rebot_pos_filter_alpha": 0.4,        # (legacy EMA; superseded by One Euro below)
        "rebot_orient_filter_alpha": 0.3,     # orientation smoothing (EMA)
        # Joint control mode: "pos_vel" (position + velocity limit; default,
        # safe) or "mit" (impedance: pos+vel reference + kp/kd + gravity
        # feedforward — smoothest/compliant, but tune gains carefully on hardware).
        "rebot_control_mode": "pos_vel",
        "rebot_mit_kp": [120, 120, 120, 18, 18, 18],
        "rebot_mit_kd": [8, 8, 8, 2, 2, 2],
        "rebot_gravity_ff": True,
        # One Euro filter on controller position: adaptive smoothing — heavy at
        # low speed (kills jitter where precision matters), light at high speed
        # (no lag). mincutoff lower = smoother when still; beta higher = snappier
        # when moving fast.
        "rebot_oneeuro_mincutoff": 1.0,
        "rebot_oneeuro_beta": 0.02,
        # Trajectory-servo Cartesian speed caps (the EE traces the path toward
        # the hand at up to these speeds; lower = smoother path, more lag).
        "rebot_max_lin_vel_m_s": 0.5,
        "rebot_max_ang_vel_deg_s": 180.0,
        # Path-following: hand pose is sampled into a waypoint queue at this
        # minimum spacing so the arm reproduces the hand's path. Smaller spacing
        # = finer path; max_waypoints caps lag behind fast hand motion.
        "rebot_waypoint_min_dist_m": 0.005,
        "rebot_waypoint_min_ang_deg": 2.0,
        "rebot_path_max_waypoints": 60,
        # Sensitivity tuning.
        #  orientation_scale: gain on controller rotation -> wrist (1.0 = 1:1,
        #    lower = wrist less sensitive to hand rotation).
        #  joint_max_vel_deg_s: per-joint speed cap [pan, lift, elbow, w_flex,
        #    w_yaw, w_roll] — lowers twitchiness of over-sensitive joints.
        "rebot_orientation_scale": 0.6,
        "rebot_joint_max_vel_deg_s": [150, 250, 250, 180, 180, 180],
        # Jerk-limited online trajectory generation (Ruckig) on joint commands —
        # the smoothness "shock absorber". Lower accel/jerk = smoother but laggier.
        # (max velocity per joint reuses rebot_joint_max_vel_deg_s above.)
        "rebot_otg_max_accel_deg_s2": 2000.0,
        "rebot_otg_max_jerk_deg_s3": 20000.0,
    },
    "control": {
        "keyboard": {
            "enabled": True,
            "pos_step": 0.01,
            "angle_step": 5.0,
            "gripper_step": 10.0
        },
        "vr": {
            "enabled": True
        },
        "pybullet": {
            "enabled": True
        }
    },
    "paths": {
        "urdf_path": "URDF/SO100/so100.urdf"
    },
    "gripper": {
        "open_angle": 0.0,
        "closed_angle": 45.0
    },
    "ik": {
        "use_reference_poses": True,
        "reference_poses_file": "reference_poses.json",
        "position_error_threshold": 0.001,
        "hysteresis_threshold": 0.01,
        "movement_penalty_weight": 0.01
    }
}

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file with fallback to defaults."""
    config = DEFAULT_CONFIG.copy()
    
    # Try to load from project root first (package installation directory)
    package_config_path = get_absolute_path(config_path)
    
    # Check if config exists in package directory
    if package_config_path.exists():
        config_file_to_use = package_config_path
        logger.info(f"Loading config from package directory: {config_file_to_use}")
    # Fallback to current working directory (for user-provided configs)
    elif os.path.exists(config_path):
        config_file_to_use = Path(config_path)
        logger.info(f"Loading config from current directory: {config_file_to_use}")
    else:
        logger.info(f"Config file {config_path} not found in package directory ({package_config_path}) or current directory, using defaults")
        return config
    
    try:
        with open(config_file_to_use, 'r') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                # Deep merge yaml config into default config
                _deep_merge(config, yaml_config)
    except Exception as e:
        logger.warning(f"Could not load config from {config_file_to_use}: {e}")
        logger.info("Using default configuration")
    
    return config

def save_config(config: dict, config_path: str = "config.yaml"):
    """Save configuration to YAML file in project root."""
    # Always save to project root directory
    abs_config_path = get_absolute_path(config_path)
    try:
        with open(abs_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving config to {abs_config_path}: {e}")
        return False

def _deep_merge(base: dict, update: dict):
    """Deep merge update dict into base dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value

# Load configuration
_config_data = load_config()

# Extract values for backward compatibility
HTTPS_PORT = _config_data["network"]["https_port"]
WEBSOCKET_PORT = _config_data["network"]["websocket_port"]
HOST_IP = _config_data["network"]["host_ip"]

CERTFILE = _config_data["ssl"]["certfile"]
KEYFILE = _config_data["ssl"]["keyfile"]

VR_TO_ROBOT_SCALE = _config_data["robot"]["vr_to_robot_scale"]
SEND_INTERVAL = _config_data["robot"]["send_interval"]
# reBot only: when False, the 6-DOF IK holds the EE orientation fixed at the
# grip-origin orientation (position-only control) — useful for safe bring-up.
# When True, the VR controller orientation drives the EE orientation.
REBOT_ORIENTATION_ENABLED = bool(_config_data["robot"].get("rebot_orientation_enabled", True))
REBOT_MOTOR_VELOCITY_DEG_S = float(_config_data["robot"].get("rebot_motor_velocity_deg_s", 250.0))
REBOT_POS_FILTER_ALPHA = float(_config_data["robot"].get("rebot_pos_filter_alpha", 0.4))
REBOT_ORIENT_FILTER_ALPHA = float(_config_data["robot"].get("rebot_orient_filter_alpha", 0.3))
REBOT_ONEEURO_MINCUTOFF = float(_config_data["robot"].get("rebot_oneeuro_mincutoff", 1.0))
REBOT_ONEEURO_BETA = float(_config_data["robot"].get("rebot_oneeuro_beta", 0.02))
REBOT_MAX_LIN_VEL_M_S = float(_config_data["robot"].get("rebot_max_lin_vel_m_s", 0.5))
REBOT_MAX_ANG_VEL_DEG_S = float(_config_data["robot"].get("rebot_max_ang_vel_deg_s", 180.0))
REBOT_WAYPOINT_MIN_DIST_M = float(_config_data["robot"].get("rebot_waypoint_min_dist_m", 0.005))
REBOT_WAYPOINT_MIN_ANG_DEG = float(_config_data["robot"].get("rebot_waypoint_min_ang_deg", 2.0))
REBOT_PATH_MAX_WAYPOINTS = int(_config_data["robot"].get("rebot_path_max_waypoints", 60))
REBOT_ORIENTATION_SCALE = float(_config_data["robot"].get("rebot_orientation_scale", 0.6))
REBOT_JOINT_MAX_VEL_DEG_S = tuple(
    float(x) for x in _config_data["robot"].get(
        "rebot_joint_max_vel_deg_s", [150, 250, 250, 180, 180, 180])
)
REBOT_OTG_MAX_ACCEL_DEG_S2 = float(_config_data["robot"].get("rebot_otg_max_accel_deg_s2", 2000.0))
REBOT_OTG_MAX_JERK_DEG_S3 = float(_config_data["robot"].get("rebot_otg_max_jerk_deg_s3", 20000.0))
REBOT_CONTROL_MODE = str(_config_data["robot"].get("rebot_control_mode", "pos_vel")).lower()
REBOT_MIT_KP = tuple(float(x) for x in _config_data["robot"].get("rebot_mit_kp", [120, 120, 120, 18, 18, 18]))
REBOT_MIT_KD = tuple(float(x) for x in _config_data["robot"].get("rebot_mit_kd", [8, 8, 8, 2, 2, 2]))
REBOT_GRAVITY_FF = bool(_config_data["robot"].get("rebot_gravity_ff", True))

POS_STEP = _config_data["control"]["keyboard"]["pos_step"]
ANGLE_STEP = _config_data["control"]["keyboard"]["angle_step"]
GRIPPER_STEP = _config_data["control"]["keyboard"]["gripper_step"]

GRIPPER_OPEN_ANGLE = _config_data["gripper"]["open_angle"]
GRIPPER_CLOSED_ANGLE = _config_data["gripper"]["closed_angle"]

# IK Configuration
USE_REFERENCE_POSES = _config_data["ik"]["use_reference_poses"]
# Reference poses file — each robot type stores its own so IK seeds match the URDF geometry
_raw_ref_file = _config_data["ik"]["reference_poses_file"]
REFERENCE_POSES_FILE = (
    "rebot_reference_poses.json"
    if _config_data["robot"].get("type", "so100").lower() == "rebot"
    else _raw_ref_file
)
IK_POSITION_ERROR_THRESHOLD = _config_data["ik"]["position_error_threshold"]
IK_HYSTERESIS_THRESHOLD = _config_data["ik"]["hysteresis_threshold"]
IK_MOVEMENT_PENALTY_WEIGHT = _config_data["ik"]["movement_penalty_weight"]

# --- Robot Type (normalised to lowercase so "reBot"/"REBOT"/"rebot" all work) ---
ROBOT_TYPE = _config_data["robot"].get("type", "so100").lower()

# --- SO100 Joint Configuration (6-DOF, Feetech STS3215 servos) ---
SO100_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SO100_NUM_JOINTS = len(SO100_JOINT_NAMES)
SO100_NUM_IK_JOINTS = 3
SO100_WRIST_FLEX_INDEX = 3
SO100_WRIST_YAW_INDEX = None   # SO100 has no wrist_yaw
SO100_WRIST_ROLL_INDEX = 4
SO100_GRIPPER_INDEX = 5
SO100_URDF_PATH = "URDF/SO100/so100.urdf"
SO100_END_EFFECTOR_LINK_NAME = "Fixed_Jaw_tip"
SO100_URDF_TO_INTERNAL_NAME_MAP = {
    "1": "shoulder_pan",
    "2": "shoulder_lift",
    "3": "elbow_flex",
    "4": "wrist_flex",
    "5": "wrist_roll",
    "6": "gripper",
}

# Motor configuration for SO100
COMMON_MOTORS = {
    "shoulder_pan": [1, "sts3215"],
    "shoulder_lift": [2, "sts3215"],
    "elbow_flex": [3, "sts3215"],
    "wrist_flex": [4, "sts3215"],
    "wrist_roll": [5, "sts3215"],
    "gripper": [6, "sts3215"],
}

# --- reBot B601-DM Joint Configuration (7-DOF, Damiao CAN motors) ---
REBOT_JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_yaw", "wrist_roll", "gripper",
]
REBOT_NUM_JOINTS = len(REBOT_JOINT_NAMES)
REBOT_NUM_IK_JOINTS = 3
REBOT_WRIST_FLEX_INDEX = 3
REBOT_WRIST_YAW_INDEX = 4   # Extra DOF not present in SO100
REBOT_WRIST_ROLL_INDEX = 5
REBOT_GRIPPER_INDEX = 6
# Official manufacturer URDF (reBot-DevArm_fixend) — accurate kinematics that
# match the real arm. The previous hand-built rebot_b601_dm.urdf was a naive
# vertical stack whose geometry did not match the hardware (caused inverted
# motion that no per-joint axis flip could fix).
REBOT_URDF_PATH = "URDF/reBot_DevArm/rebot_devarm.urdf"
REBOT_END_EFFECTOR_LINK_NAME = "end_link"
# Maps the manufacturer URDF joint names -> telegrip internal joint names.
# joint1..joint6 are the 6 arm DOF in base->tip order; "join3" is a typo in the
# manufacturer URDF (kept verbatim so the lookup matches). The gripper joint we
# appended to the URDF keeps its own name.
REBOT_URDF_TO_INTERNAL_NAME_MAP = {
    "joint1": "shoulder_pan",
    "joint2": "shoulder_lift",
    "join3":  "elbow_flex",     # NOTE: manufacturer URDF misspells joint3 as "join3"
    "joint4": "wrist_flex",
    "joint5": "wrist_yaw",
    "joint6": "wrist_roll",
    "gripper": "gripper",
}
# Default home positions (degrees).
# SO100: a known safe resting pose.
SO100_HOME_POSITION = [0, -100, 100, 60, 0, 0]
# reBot: the official SDK resets every motor's zero to the current physical pose
# at startup, so all joints read 0° after power-on calibration.
REBOT_HOME_POSITION = [0, 0, 0, 0, 0, 0, 0]

# --- Active robot constants (selected by robot_type) ---
if ROBOT_TYPE == "rebot":
    JOINT_NAMES = REBOT_JOINT_NAMES
    NUM_JOINTS = REBOT_NUM_JOINTS
    NUM_IK_JOINTS = REBOT_NUM_IK_JOINTS
    WRIST_FLEX_INDEX = REBOT_WRIST_FLEX_INDEX
    WRIST_YAW_INDEX = REBOT_WRIST_YAW_INDEX
    WRIST_ROLL_INDEX = REBOT_WRIST_ROLL_INDEX
    GRIPPER_INDEX = REBOT_GRIPPER_INDEX
    URDF_TO_INTERNAL_NAME_MAP = REBOT_URDF_TO_INTERNAL_NAME_MAP
    END_EFFECTOR_LINK_NAME = REBOT_END_EFFECTOR_LINK_NAME
    if _config_data["paths"]["urdf_path"] == SO100_URDF_PATH:
        # Auto-select reBot URDF when user hasn't overridden the path
        URDF_PATH = REBOT_URDF_PATH
    else:
        URDF_PATH = _config_data["paths"]["urdf_path"]
else:
    JOINT_NAMES = SO100_JOINT_NAMES
    NUM_JOINTS = SO100_NUM_JOINTS
    NUM_IK_JOINTS = SO100_NUM_IK_JOINTS
    WRIST_FLEX_INDEX = SO100_WRIST_FLEX_INDEX
    WRIST_YAW_INDEX = SO100_WRIST_YAW_INDEX
    WRIST_ROLL_INDEX = SO100_WRIST_ROLL_INDEX
    GRIPPER_INDEX = SO100_GRIPPER_INDEX
    URDF_TO_INTERNAL_NAME_MAP = SO100_URDF_TO_INTERNAL_NAME_MAP
    END_EFFECTOR_LINK_NAME = SO100_END_EFFECTOR_LINK_NAME
    URDF_PATH = _config_data["paths"]["urdf_path"]

# --- Keyboard Control ---
POS_STEP = 0.01  # meters
ANGLE_STEP = 5.0 # degrees
GRIPPER_STEP = 10.0 # degrees

# --- Device Ports ---
DEFAULT_FOLLOWER_PORTS = {
    "left": _config_data["robot"]["left_arm"]["port"],
    "right": _config_data["robot"]["right_arm"]["port"]
}

@dataclass
class TelegripConfig:
    """Main configuration class for the teleoperation system."""
    
    # Network settings
    https_port: int = HTTPS_PORT
    websocket_port: int = WEBSOCKET_PORT
    host_ip: str = HOST_IP
    
    # SSL settings
    certfile: str = CERTFILE
    keyfile: str = KEYFILE
    
    # Robot settings
    vr_to_robot_scale: float = VR_TO_ROBOT_SCALE
    send_interval: float = SEND_INTERVAL
    rebot_orientation_enabled: bool = REBOT_ORIENTATION_ENABLED
    rebot_motor_velocity_deg_s: float = REBOT_MOTOR_VELOCITY_DEG_S
    rebot_pos_filter_alpha: float = REBOT_POS_FILTER_ALPHA
    rebot_orient_filter_alpha: float = REBOT_ORIENT_FILTER_ALPHA
    rebot_oneeuro_mincutoff: float = REBOT_ONEEURO_MINCUTOFF
    rebot_oneeuro_beta: float = REBOT_ONEEURO_BETA
    rebot_max_lin_vel_m_s: float = REBOT_MAX_LIN_VEL_M_S
    rebot_max_ang_vel_deg_s: float = REBOT_MAX_ANG_VEL_DEG_S
    rebot_waypoint_min_dist_m: float = REBOT_WAYPOINT_MIN_DIST_M
    rebot_waypoint_min_ang_deg: float = REBOT_WAYPOINT_MIN_ANG_DEG
    rebot_path_max_waypoints: int = REBOT_PATH_MAX_WAYPOINTS
    rebot_orientation_scale: float = REBOT_ORIENTATION_SCALE
    rebot_joint_max_vel_deg_s: tuple = REBOT_JOINT_MAX_VEL_DEG_S
    rebot_otg_max_accel_deg_s2: float = REBOT_OTG_MAX_ACCEL_DEG_S2
    rebot_otg_max_jerk_deg_s3: float = REBOT_OTG_MAX_JERK_DEG_S3
    rebot_control_mode: str = REBOT_CONTROL_MODE
    rebot_mit_kp: tuple = REBOT_MIT_KP
    rebot_mit_kd: tuple = REBOT_MIT_KD
    rebot_gravity_ff: bool = REBOT_GRAVITY_FF
    
    # Device ports
    follower_ports: Dict[str, str] = None
    arm_enabled: Dict[str, bool] = None
    calibration_paths: Dict[str, Optional[str]] = None
    
    # Robot type
    robot_type: str = ROBOT_TYPE

    # Control flags
    enable_pybullet: bool = True
    enable_pybullet_gui: bool = True
    enable_robot: bool = True
    enable_vr: bool = True
    enable_keyboard: bool = True
    autoconnect: bool = False
    log_level: str = "warning"

    # Paths
    urdf_path: str = URDF_PATH
    webapp_dir: str = "webapp"
    
    # IK settings
    use_reference_poses: bool = USE_REFERENCE_POSES
    reference_poses_file: str = REFERENCE_POSES_FILE
    ik_position_error_threshold: float = IK_POSITION_ERROR_THRESHOLD
    ik_hysteresis_threshold: float = IK_HYSTERESIS_THRESHOLD
    ik_movement_penalty_weight: float = IK_MOVEMENT_PENALTY_WEIGHT
    
    # Gripper settings
    gripper_open_angle: float = GRIPPER_OPEN_ANGLE
    gripper_closed_angle: float = GRIPPER_CLOSED_ANGLE
    
    # Keyboard control
    pos_step: float = POS_STEP
    angle_step: float = ANGLE_STEP
    gripper_step: float = GRIPPER_STEP
    
    def __post_init__(self):
        # Initialize follower_ports if not set
        if self.follower_ports is None:
            self.follower_ports = {
                "left": _config_data["robot"]["left_arm"]["port"],
                "right": _config_data["robot"]["right_arm"]["port"]
            }

        if self.arm_enabled is None:
            self.arm_enabled = {
                "left": _config_data["robot"]["left_arm"].get("enabled", True),
                "right": _config_data["robot"]["right_arm"].get("enabled", True)
            }

        if self.calibration_paths is None:
            def _expand(p):
                return str(Path(p).expanduser()) if p else None
            self.calibration_paths = {
                "left":  _expand(_config_data["robot"]["left_arm"].get("calibration")),
                "right": _expand(_config_data["robot"]["right_arm"].get("calibration")),
            }
        
        # Ensure ports are not None
        if self.follower_ports["left"] is None:
            self.follower_ports["left"] = "/dev/ttyACM0"
        if self.follower_ports["right"] is None:
            self.follower_ports["right"] = "/dev/ttyACM1"
    
    @property
    def is_rebot(self) -> bool:
        return self.robot_type == "rebot"

    @property
    def num_joints(self) -> int:
        return REBOT_NUM_JOINTS if self.is_rebot else SO100_NUM_JOINTS

    @property
    def joint_names(self) -> list:
        return REBOT_JOINT_NAMES if self.is_rebot else SO100_JOINT_NAMES

    @property
    def wrist_flex_index(self) -> int:
        return REBOT_WRIST_FLEX_INDEX if self.is_rebot else SO100_WRIST_FLEX_INDEX

    @property
    def wrist_yaw_index(self) -> Optional[int]:
        return REBOT_WRIST_YAW_INDEX if self.is_rebot else SO100_WRIST_YAW_INDEX

    @property
    def wrist_roll_index(self) -> int:
        return REBOT_WRIST_ROLL_INDEX if self.is_rebot else SO100_WRIST_ROLL_INDEX

    @property
    def gripper_index(self) -> int:
        return REBOT_GRIPPER_INDEX if self.is_rebot else SO100_GRIPPER_INDEX

    @property
    def home_position(self) -> list:
        return REBOT_HOME_POSITION if self.is_rebot else SO100_HOME_POSITION

    @property
    def ssl_files_exist(self) -> bool:
        """Check if SSL certificate files exist."""
        cert_path = get_absolute_path(self.certfile)
        key_path = get_absolute_path(self.keyfile)
        return cert_path.exists() and key_path.exists()
    
    def ensure_ssl_certificates(self) -> bool:
        """Ensure SSL certificates exist, generating them if necessary."""
        from .utils import ensure_ssl_certificates
        return ensure_ssl_certificates(self.certfile, self.keyfile)
    
    @property
    def urdf_exists(self) -> bool:
        """Check if URDF file exists."""
        urdf_path = get_absolute_path(self.urdf_path)
        return urdf_path.exists()
    
    @property
    def webapp_exists(self) -> bool:
        """Check if webapp directory exists."""
        webapp_path = get_absolute_path(self.webapp_dir)
        return webapp_path.exists()
    
    def get_absolute_urdf_path(self) -> str:
        """Get absolute path to URDF file."""
        return str(get_absolute_path(self.urdf_path))
    
    def get_absolute_reference_poses_path(self) -> str:
        """Get absolute path to reference poses file."""
        return str(get_absolute_path(self.reference_poses_file))
    
    def get_absolute_ssl_paths(self) -> tuple:
        """Get absolute paths to SSL certificate files."""
        cert_path = str(get_absolute_path(self.certfile))
        key_path = str(get_absolute_path(self.keyfile))
        return cert_path, key_path

def get_config_data():
    """Get the current configuration data."""
    return _config_data.copy()

def update_config_data(new_config: dict):
    """Update the global configuration data."""
    global _config_data
    _config_data = new_config
    
    # Save to file
    save_config(_config_data)

# Global configuration instance
config = TelegripConfig() 
