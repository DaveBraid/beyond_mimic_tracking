"""在 Isaac Lab 中检查机器人是否存在自碰撞。

示例：

    python scripts/check_self_collision.py --robot lrs
    python scripts/check_self_collision.py --robot g1 --mode static
    python scripts/check_self_collision.py --robot s3 --mode sweep --headless
"""

import argparse
import math

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="启动 Isaac Lab 仿真并检测指定机器人是否发生自碰撞。",
    epilog=(
        "说明：\n"
        "  1. 脚本默认不放地面，并关闭机器人重力，尽量只观察机器人自身的碰撞。\n"
        "  2. `static` 模式保持默认姿态；`sweep` 模式会在关节限位内做小幅往复运动。\n"
        "  3. 输出中的 `bodies` 表示当前检测到接触力超过阈值的机器人刚体名称。"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--robot", type=str, choices=["lrs", "g1", "s3"], help="要加载的机器人类型。")
parser.add_argument(
    "--mode",
    type=str,
    choices=["static", "sweep"],
    default="sweep",
    help="检测模式：`static` 保持默认姿态，`sweep` 在关节范围内做正弦扫动。",
)
parser.add_argument("--num_steps", type=int, default=0, help="总共运行的物理步数；设为 0 表示持续运行直到手动关闭。")
parser.add_argument("--threshold", type=float, default=1.0, help="判定为碰撞的接触力阈值，单位牛顿。")
parser.add_argument(
    "--amplitude_scale",
    type=float,
    default=0.20,
    help="`sweep` 模式下使用的关节扫动幅度，占关节活动范围的比例。",
)
parser.add_argument("--frequency", type=float, default=0.5, help="`sweep` 模式下的扫动频率，单位 Hz。")
parser.add_argument("--print_interval", type=int, default=50, help="日志打印间隔，单位为仿真步。")
parser.add_argument(
    "--stop_on_first_collision",
    action="store_true",
    help="一旦检测到自碰撞，立即停止仿真。",
)
parser.add_argument("--show_ground", action="store_true", help="显示地面，便于观察姿态和空间位置。")
parser.add_argument("--show_collision_markers", action="store_true", help="显示 Isaac Lab 的接触力可视化标记。")
parser.add_argument("--camera_follow", action="store_true", help="让相机持续跟随机器人根部。")
parser.add_argument("--camera_distance", type=float, default=2.0, help="相机相对机器人的水平观察距离。")
parser.add_argument("--camera_height", type=float, default=1.0, help="相机相对机器人的竖直观察高度。")
parser.add_argument("--enable_gravity", action="store_true", help="开启重力，让机器人按真实物理响应。")
parser.add_argument(
    "--free_root",
    action="store_true",
    help="不在每一步强制写回 root state，让机器人根部可以自由运动。",
)
parser.add_argument(
    "--freeze_joints",
    action="store_true",
    help="不发送关节位置目标，让机器人只受当前动力学与重力影响。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
APP_LIVESTREAM = getattr(args_cli, "livestream", -1)

if args_cli.robot is None:
    parser.error("参数 `--robot` 是必需的，可选值：lrs / g1 / s3")

if APP_LIVESTREAM is not None and APP_LIVESTREAM >= 0:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
SHOULD_RENDER = (not args_cli.headless) or (APP_LIVESTREAM is not None and APP_LIVESTREAM >= 0)
print("[DEBUG]: AppLauncher finished.", flush=True)
print(
    f"[DEBUG]: headless={args_cli.headless}, livestream={APP_LIVESTREAM}, "
    f"enable_cameras={getattr(args_cli, 'enable_cameras', None)}, should_render={SHOULD_RENDER}, device={args_cli.device}",
    flush=True,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
print("[DEBUG]: Isaac Lab imports finished.", flush=True)

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.robots.lrs import LRS_CYLINDER_CFG
from whole_body_tracking.robots.shen_nong3 import S3_CYLINDER_CFG


def _get_robot_cfg(name: str) -> ArticulationCfg:
    if name == "g1":
        base_cfg = G1_CYLINDER_CFG
    elif name == "lrs":
        base_cfg = LRS_CYLINDER_CFG
    elif name == "s3":
        base_cfg = S3_CYLINDER_CFG
    else:
        raise ValueError(f"Unsupported robot: {name}")

    return base_cfg.replace(
        spawn=base_cfg.spawn.replace(
            rigid_props=base_cfg.spawn.rigid_props.replace(disable_gravity=not args_cli.enable_gravity),
        )
    )


ROBOT_CFG = _get_robot_cfg(args_cli.robot)


@configclass
class SelfCollisionSceneCfg(InteractiveSceneCfg):
    if args_cli.show_ground:
        ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=args_cli.threshold,
        debug_vis=args_cli.show_collision_markers,
    )


def _build_targets(robot: Articulation, sim_time: float) -> torch.Tensor:
    default_joint_pos = robot.data.default_joint_pos.clone()
    if args_cli.mode == "static":
        return default_joint_pos

    lower = robot.data.soft_joint_pos_limits[..., 0]
    upper = robot.data.soft_joint_pos_limits[..., 1]
    joint_range = upper - lower
    amplitude = 0.5 * args_cli.amplitude_scale * joint_range
    phase = torch.linspace(0.0, math.pi, robot.num_joints, device=robot.device).unsqueeze(0)
    signal = torch.sin(2.0 * math.pi * args_cli.frequency * sim_time + phase)
    targets = default_joint_pos + amplitude * signal
    return torch.clamp(targets, lower, upper)


def main():
    print("[INFO]: App launch config")
    print(f"  headless={args_cli.headless}")
    print(f"  livestream={APP_LIVESTREAM}")
    print(f"  enable_cameras={getattr(args_cli, 'enable_cameras', None)}")
    print(f"  should_render={SHOULD_RENDER}")
    print(f"  device={args_cli.device}")

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene = InteractiveScene(SelfCollisionSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()

    robot: Articulation = scene["robot"]
    contact_sensor = scene["contact_forces"]

    root_state = robot.data.default_root_state.clone()
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()
    scene.write_data_to_sim()
    sim.step(render=False)
    if SHOULD_RENDER:
        sim.render()
    scene.update(sim.get_physics_dt())

    base_pos = robot.data.root_pos_w[0].cpu().numpy()
    sim.set_camera_view(
        base_pos + (args_cli.camera_distance, args_cli.camera_distance, args_cli.camera_height),
        base_pos,
    )

    print(f"[INFO]: Robot: {args_cli.robot}")
    print(f"[INFO]: Bodies monitored: {len(contact_sensor.body_names)}")
    print(f"[INFO]: Mode: {args_cli.mode}")
    print(f"[INFO]: Ground: {'on' if args_cli.show_ground else 'off'}")
    print(f"[INFO]: Collision markers: {'on' if args_cli.show_collision_markers else 'off'}")
    print(f"[INFO]: Gravity: {'on' if args_cli.enable_gravity else 'off'}")
    print(f"[INFO]: Free root: {'on' if args_cli.free_root else 'off'}")
    print(f"[INFO]: Freeze joints: {'on' if args_cli.freeze_joints else 'off'}")

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    any_collision = False
    previous_collision_bodies: tuple[str, ...] = tuple()

    step = 0
    while simulation_app.is_running():
        if args_cli.num_steps > 0 and step >= args_cli.num_steps:
            break
        targets = _build_targets(robot, sim_time)
        if not args_cli.free_root:
            robot.write_root_state_to_sim(root_state)
        if not args_cli.freeze_joints:
            robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step(render=False)
        if SHOULD_RENDER:
            sim.render()
        scene.update(sim_dt)
        sim_time += sim_dt

        net_forces = contact_sensor.data.net_forces_w[0]
        force_norm = torch.norm(net_forces, dim=-1)
        colliding_ids = torch.nonzero(force_norm > args_cli.threshold, as_tuple=False).squeeze(-1)

        if colliding_ids.numel() > 0:
            any_collision = True
            colliding_bodies = tuple(contact_sensor.body_names[idx] for idx in colliding_ids.tolist())
            max_force = force_norm[colliding_ids].max().item()
            should_print = (
                colliding_bodies != previous_collision_bodies
                or step % args_cli.print_interval == 0
                or step == 0
            )
            if should_print:
                print(f"[COLLISION][step={step:04d}][t={sim_time:.3f}s] max_force={max_force:.3f}N")
                print("  bodies:", ", ".join(colliding_bodies))
            previous_collision_bodies = colliding_bodies
            if args_cli.stop_on_first_collision:
                break
        elif step % args_cli.print_interval == 0:
            print(f"[OK][step={step:04d}][t={sim_time:.3f}s] no self-collision above {args_cli.threshold:.3f}N")
            previous_collision_bodies = tuple()

        if SHOULD_RENDER and args_cli.camera_follow:
            base_pos = robot.data.root_pos_w[0].cpu().numpy()
            sim.set_camera_view(
                base_pos + (args_cli.camera_distance, args_cli.camera_distance, args_cli.camera_height), base_pos
            )

        step += 1

    if any_collision:
        print("[RESULT]: Self-collision detected.")
    else:
        print("[RESULT]: No self-collision detected above threshold.")


if __name__ == "__main__":
    main()
    simulation_app.close()
