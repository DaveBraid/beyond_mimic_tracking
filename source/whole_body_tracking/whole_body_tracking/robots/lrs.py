import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

LRS_URDF_PATH = f"{ASSET_DIR}/lrs/new/urdf/LRS-XURDF0210QJ.urdf"

ARMATURE_58P = 0.002746621
ARMATURE_88P_14 = 0.041124596
ARMATURE_88P_22 = 0.016712311

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_58P = ARMATURE_58P * NATURAL_FREQ**2
STIFFNESS_88P_14 = ARMATURE_88P_14 * NATURAL_FREQ**2
STIFFNESS_88P_22 = ARMATURE_88P_22 * NATURAL_FREQ**2

DAMPING_58P = 2.0 * DAMPING_RATIO * ARMATURE_58P * NATURAL_FREQ
DAMPING_88P_14 = 2.0 * DAMPING_RATIO * ARMATURE_88P_14 * NATURAL_FREQ
DAMPING_88P_22 = 2.0 * DAMPING_RATIO * ARMATURE_88P_22 * NATURAL_FREQ

LRS_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_joint",
    "left_hand_pitch_joint",
    "left_hand_roll_joint",
    "left_hand_yaw_joint",
    "left_hand_elbow_joint",
    "left_hand_wrist_joint",
    "right_hand_pitch_joint",
    "right_hand_roll_joint",
    "right_hand_yaw_joint",
    "right_hand_elbow_joint",
    "right_hand_wrist_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

LRS_CSV_JOINT_SIGNS = [1.0] * len(LRS_JOINT_NAMES)

LRS_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        asset_path=LRS_URDF_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            "left_hand_roll_joint": -0.20,
            "right_hand_roll_joint": -0.20,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint"],
            effort_limit_sim=25.0,
            velocity_limit_sim=31.415927,
            stiffness=STIFFNESS_58P,
            damping=DAMPING_58P,
            armature=ARMATURE_58P,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 140.0,
                ".*_hip_roll_joint": 90.0,
                ".*_hip_yaw_joint": 90.0,
                ".*_knee_joint": 140.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 20.943951,
                ".*_hip_roll_joint": 32.986723,
                ".*_hip_yaw_joint": 32.986723,
                ".*_knee_joint": 20.943951,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_88P_22,
                ".*_hip_roll_joint": STIFFNESS_88P_14,
                ".*_hip_yaw_joint": STIFFNESS_88P_14,
                ".*_knee_joint": STIFFNESS_88P_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_88P_22,
                ".*_hip_roll_joint": DAMPING_88P_14,
                ".*_hip_yaw_joint": DAMPING_88P_14,
                ".*_knee_joint": DAMPING_88P_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_88P_22,
                ".*_hip_roll_joint": ARMATURE_88P_14,
                ".*_hip_yaw_joint": ARMATURE_88P_14,
                ".*_knee_joint": ARMATURE_88P_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=25.0,
            velocity_limit_sim=31.415927,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=STIFFNESS_58P,
            damping=DAMPING_58P,
            armature=ARMATURE_58P,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=25.0,
            velocity_limit_sim=31.415927,
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
            stiffness=STIFFNESS_58P,
            damping=DAMPING_58P,
            armature=ARMATURE_58P,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=90.0,
            velocity_limit_sim=32.986723,
            stiffness=STIFFNESS_88P_14,
            damping=DAMPING_88P_14,
            armature=ARMATURE_88P_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_pitch_joint",
                ".*_hand_roll_joint",
                ".*_hand_yaw_joint",
                ".*_hand_elbow_joint",
                ".*_hand_wrist_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=31.415927,
            stiffness=STIFFNESS_58P,
            damping=DAMPING_58P,
            armature=ARMATURE_58P,
        ),
    },
)

LRS_ACTION_SCALE = {}
for actuator in LRS_CYLINDER_CFG.actuators.values():
    effort_limit = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    joint_names = actuator.joint_names_expr
    if not isinstance(effort_limit, dict):
        effort_limit = {name: effort_limit for name in joint_names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in joint_names}
    for name in joint_names:
        if name in effort_limit and name in stiffness and stiffness[name]:
            LRS_ACTION_SCALE[name] = 0.25 * effort_limit[name] / stiffness[name]
