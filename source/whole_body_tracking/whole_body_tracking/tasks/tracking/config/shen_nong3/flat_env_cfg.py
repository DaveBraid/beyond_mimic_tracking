from isaaclab.utils import configclass

from whole_body_tracking.robots.shen_nong3 import S3_ACTION_SCALE, S3_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.shen_nong3.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class S3FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = S3_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = S3_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_foot_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_foot_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_hand_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_hand_link",
        ]


@configclass
class S3FlatWoStateEstimationEnvCfg(S3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class S3FlatLowFreqEnvCfg(S3FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
