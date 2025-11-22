from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def gate_dof_armature_by_episode_length(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    target_mean_len: float, 
    active_range: tuple[float, float]
):
    """
    根据平均回合长度开启 DoF Armature 随机化。
    """
    # 1. 初始化统计追踪器
    if not hasattr(env, "curriculum_tracker"):
        env.curriculum_tracker = {
            "total_len": 0.0,
            "num_episodes": 0,
            "triggered_events": set()
        }
    
    tracker = env.curriculum_tracker

    # 2. 如果已经触发了课程，直接返回
    if "armature" in tracker.get("triggered_events", set()):
        return

    # 3. 统计已完成回合的长度
    # 错误写法: dones = env.reset_buf
    # 正确写法: 直接使用 env_ids，它包含了当前正在重置的环境索引
    if len(env_ids) > 0:
        finished_lengths = env.episode_length_buf[env_ids]
        
        tracker["total_len"] += torch.sum(finished_lengths).item()
        tracker["num_episodes"] += len(finished_lengths)
        
        # 防止除以零
        if tracker["num_episodes"] == 0:
            return

        # 计算当前的平均值
        mean_len = tracker["total_len"] / tracker["num_episodes"]
        
        # 4. 检查是否满足阈值
        if mean_len >= target_mean_len:
            print(f"\n[Curriculum] Mean Episode Length ({mean_len:.1f}) >= {target_mean_len}. Enabling Armature Randomization!")
            
            try:
                event_term = env.event_manager._terms["add_dof_armature"]
                event_term.cfg.params["armature_distribution_params"] = active_range
                
                # 记录触发状态
                if "triggered_events" not in tracker: tracker["triggered_events"] = set()
                tracker["triggered_events"].add("armature")
            except KeyError:
                print("[Curriculum] Warning: Event term 'add_dof_armature' not found.")


def gate_push_velocity_by_episode_length(
    env: ManagerBasedRLEnv, 
    env_ids: torch.Tensor, 
    target_mean_len: float, 
    new_velocity_range: dict[str, tuple[float, float]]
):
    """
    当平均回合长度达到阈值时，增大推力随机化范围。
    """
    if not hasattr(env, "curriculum_tracker"):
        env.curriculum_tracker = {
            "total_len": 0.0, "num_episodes": 0, 
            "triggered_events": set() 
        }
    
    tracker = env.curriculum_tracker
    
    if "push_increase" in tracker.get("triggered_events", set()):
        return

    # 同样这里也只需要使用 env_ids
    if len(env_ids) > 0:
        finished_lengths = env.episode_length_buf[env_ids]
        tracker["total_len"] += torch.sum(finished_lengths).item()
        tracker["num_episodes"] += len(finished_lengths)
    
    if tracker["num_episodes"] == 0:
        return

    mean_len = tracker["total_len"] / tracker["num_episodes"]

    if mean_len >= target_mean_len:
        print(f"\n[Curriculum] Mean Episode Length ({mean_len:.1f}) >= {target_mean_len}. INCREASING PUSH VELOCITY!")
        
        try:
            event_term = env.event_manager._terms["push_robot"]
            event_term.cfg.params["velocity_range"] = new_velocity_range
            
            if "triggered_events" not in tracker: tracker["triggered_events"] = set()
            tracker["triggered_events"].add("push_increase")
            
        except KeyError:
            print("[Curriculum] Warning: Event term 'push_robot' not found.")