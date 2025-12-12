from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_dof_armature(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    armature_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    '''
    随机化指定DOF的armature值, 以在不测定电枢时进行to real.
    '''

    # 提取资产品 (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # 解析环境ID
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 解析关节ID
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes  优化：使用切片
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if armature_distribution_params is not None:
        # 获取当前的armature值
        arma_data = asset.data.default_joint_armature.clone()  # (num_envs, num_dofs)

        # 存储标称值以供导出
        if not hasattr(asset.data, "default_dof_armature_nominal"):
            asset.data.default_dof_armature_nominal = arma_data[0].clone()

        randomize_arma = _randomize_prop_by_op(
            arma_data,
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]

        # 更新字典属性
        # if env_ids != slice(None) and joint_ids != slice(None):
        #     env_ids = env_ids[:, None]
        asset.write_joint_armature_to_sim(randomize_arma, joint_ids, env_ids)


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains (stiffness and damping)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # -- Stiffness (Kp)
    if stiffness_distribution_params is not None:
        # get nominal values
        stiffness = asset.data.default_stiffness.clone()
        # save nominal value for export
        if not hasattr(asset.data, "default_stiffness_nominal"):
            asset.data.default_stiffness_nominal = stiffness[0].clone()
            
        # randomize
        stiffness = _randomize_prop_by_op(
            stiffness,
            stiffness_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]
        
        # apply
        asset.write_joint_stiffness_to_sim(stiffness, joint_ids, env_ids)

    # -- Damping (Kd)
    if damping_distribution_params is not None:
        # get nominal values
        damping = asset.data.default_damping.clone()
        # save nominal value for export
        if not hasattr(asset.data, "default_damping_nominal"):
            asset.data.default_damping_nominal = damping[0].clone()

        # randomize
        damping = _randomize_prop_by_op(
            damping,
            damping_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )[env_ids][:, joint_ids]
        
        # apply
        asset.write_joint_damping_to_sim(damping, joint_ids, env_ids)

