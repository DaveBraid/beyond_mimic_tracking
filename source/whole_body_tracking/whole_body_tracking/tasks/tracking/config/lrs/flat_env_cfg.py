from isaaclab.utils import configclass

from whole_body_tracking.robots.lrs import LRS_ACTION_SCALE, LRS_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.lrs.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg_lrs import TrackingEnvCfgLRS


@configclass
class LRSFlatEnvCfg(TrackingEnvCfgLRS):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = LRS_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = LRS_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "torso_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_hand_roll_link",
            "left_hand_elbow_link",
            "left_hand_wrist_link",
            "right_hand_roll_link",
            "right_hand_elbow_link",
            "right_hand_wrist_link",
        ]


@configclass
class LRSFlatWoStateEstimationEnvCfg(LRSFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class LRSFlatLowFreqEnvCfg(LRSFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
