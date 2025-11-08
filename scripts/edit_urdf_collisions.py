"""
此脚本用于将 URDF 文件加载到 Isaac Lab 交互式场景中，
以便进行物理仿真、碰撞可视化和交互式编辑。

.. code-block:: bash

    # 用法
    # 确保先启动 Isaac Sim
    # 例如:
    # ./isaac-sim.sh -r scripts/edit_urdf_collisions.py --urdf_path /path/to/your/robot.urdf
    
    # 如果你使用的是 conda 安装的 isaaclab
    # python scripts/edit_urdf_collisions.py --urdf_path /path/to/your/robot.urdf
"""

"""启动 Isaac Sim 模拟器"""

import argparse
import numpy as np
import os

from isaaclab.app import AppLauncher

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="在 Isaac Lab 中交互式加载 URDF 文件。")
parser.add_argument("--urdf_path", type=str, required=True, help="要加载的 URDF 文件的绝对路径。")
parser.add_argument("--num_envs", type=int, default=1, help="要创建的环境数量。")

# 附加 AppLauncher cli 参数
AppLauncher.add_app_launcher_args(parser)
# 解析参数
args_cli = parser.parse_args()

# 启动 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其他一切照旧"""

import torch

import isaaclab.sim as sim_utils
# 修正 1: 从 actuators 导入
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# 预定义配置
##

@configclass
class InteractiveUrdfSceneCfg(InteractiveSceneCfg):
    """配置一个包含机器人和地面平面的场景。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg()
    )

    # 灯光
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # 机器人
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=args_cli.urdf_path,  # 从命令行参数获取路径
            activate_contact_sensors=True,
            replace_cylinders_with_capsules=False, # 默认保留原始碰撞体
            fix_base=False, # 允许机器人在物理作用下移动
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            
            # 修正 2: 添加缺失的 joint_drive 配置
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),  # 将机器人放置在地面上方
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg( # 修正 1 (已保留)
                joint_names_expr=[".*"], # 为所有关节启用驱动
                stiffness=1000.0,
                damping=100.0,
            )
        }
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行模拟循环。"""
    
    # 定义模拟步进
    sim_dt = sim.get_physics_dt()
    
    # 仿真循环
    while simulation_app.is_running():
        # 检查机器人是否已初始化
        if not scene["robot"].is_initialized:
            # 等待机器人初始化
            scene.update(sim_dt)
            continue
            
        # 运行物理仿真
        sim.step()
        
        # 更新场景状态
        scene.update(sim_dt)
        
        # 将摄像机聚焦到机器人
        # （可选，你也可以在UI中手动导航）
        try:
            robot_pos = scene["robot"].data.root_pos_w[0].cpu().numpy()
            sim.set_camera_view(robot_pos + np.array([3.0, 3.0, 2.0]), robot_pos)
        except Exception:
            # 如果机器人数据还不可用，则跳过
            pass


def main():
    """主函数"""
    
    # 检查 URDF 路径是否存在
    if not os.path.isabs(args_cli.urdf_path):
        print(f"错误：URDF 路径 '{args_cli.urdf_path}' 不是绝对路径。请提供绝对路径。")
        app_launcher.close()
        return
    if not os.path.exists(args_cli.urdf_path):
        print(f"错误：URDF 文件 '{args_cli.urdf_path}' 不存在。")
        app_launcher.close()
        return

    # 加载 Sim 助手
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=0.01) # 使用 100Hz 仿真
    sim = SimulationContext(sim_cfg)
    
    # 设计场景
    scene_cfg = InteractiveUrdfSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # 运行模拟器
    sim.reset()
    
    # 现在我们准备好了！
    print("[INFO]: 设置完成。正在启动交互式仿真...")
    
    # 运行模拟器
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        # 运行主函数
        main()
    except Exception as e:
        print(f"运行出错: {e}")
    finally:
        # 关闭 sim app
        simulation_app.close()