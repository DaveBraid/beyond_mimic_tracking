"""
RSL-RL 智能选择器模块 (V9 - 历史模型版)
在手动选择流程中，增加 "自动选择最新" 或 "手动选择历史模型" 的选项。

特性:
1. "快速启动" 保持不变 (自动加载全局最新模型)。
2. "手动选择" 流程在选定 Time/Run 目录后：
    a. 显示该目录下的 "最新模型" (推荐)。
    b. 提供 "手动选择历史模型" 选项。
3. 封装为 `SmartPlaySelector` 类，可通过 `require_motion_file` 参数配置。
4. 自动过滤动作文件，只显示每个动作的最新 ':vN' 版本。
5. 在选择日期时提供 'run_name' 预览。
6. 使用 `simple-term-menu` 提供交互式光标菜单。
"""

import os
import sys
import re
from pathlib import Path

# --- 依赖检查已移入构造函数 ---
try:
    from simple_term_menu import TerminalMenu
except ImportError:
    print("错误：本项目需要 'simple-term-menu' 库。")
    print("请在您的 Python 环境中运行 'pip install simple-term-menu' 来安装它。")
    sys.exit(1)

class SmartPlaySelector:
    """
    一个可配置的终端菜单选择器，用于选择任务、动作文件、最新的检查点和环境数量。
    """
    def __init__(self, require_motion_file: bool = True, default_num_envs: int = 2):
        """
        初始化选择器。
        
        参数:
            require_motion_file (bool): 是否需要“选择动作文件”这一步。
            default_num_envs (int): 默认显示的环境数量。
        """
        self.TerminalMenu = TerminalMenu
        self.require_motion_file = require_motion_file
        self.project_root = Path(__file__).parent.parent.parent
        self.default_num_envs = default_num_envs
        
        # 预定义配置
        self.TASKS = [
            "Tracking-Flat-S3-Wo-State-Estimation-v0",
            "Tracking-Flat-G1-Wo-State-Estimation-v0",
            "Tracking-Flat-S3-v0",
            "Tracking-Flat-G1-v0",
            "Tracking-Flat-Walk-Humanoid-v0",
            "Tracking-Flat-WalkBack-Humanoid-v0",
            "Tracking-Flat-WalkBox-Humanoid-v0",
        ]
        self.TASK_LOG_MAP = {
            "Tracking-Flat-G1-Wo-State-Estimation-v0": "g1_flat",
            "Tracking-Flat-G1-v0": "g1_flat",
            "Tracking-Flat-S3-Wo-State-Estimation-v0": "s3_flat",
            "Tracking-Flat-S3-v0": "s3_flat",
            "Tracking-Flat-Walk-Humanoid-v0": "humanoid_flat",
            "Tracking-Flat-WalkBack-Humanoid-v0": "humanoid_flat",
            "Tracking-Flat-WalkBox-Humanoid-v0": "humanoid_flat",
        }
        
        self.REVERSE_LOG_MAP = {}
        self._build_reverse_log_map()
        
        self.latest_checkpoint_info: tuple[Path, str] | None = None # (path, log_dir_name)
        self._find_latest_checkpoint()

        # 结果
        self.selected_task: str = None
        self.selected_motion_file: Path = None
        self.selected_checkpoint_file: Path = None
        self.num_envs: int = self.default_num_envs
        self.logs_base_dir: Path = None 

    # --- 快速启动辅助函数 ---

    def _build_reverse_log_map(self):
        """
        从 TASK_LOG_MAP 创建一个反向映射 (例如 'g1_flat' -> ['Task1', 'Task2'])。
        """
        for task, log_dir in self.TASK_LOG_MAP.items():
            if log_dir not in self.REVERSE_LOG_MAP:
                self.REVERSE_LOG_MAP[log_dir] = []
            self.REVERSE_LOG_MAP[log_dir].append(task)

    def _find_latest_checkpoint(self):
        """
        扫描 *所有* 日志目录，找到修改时间最新的 'model_*.pt' 文件。
        """
        log_root = self.project_root / "logs" / "rsl_rl"
        if not log_root.exists():
            print(f"[警告] 'logs/rsl_rl' 目录未找到。无法启用 '快速启动'。")
            return

        print(f"正在扫描 {log_root.relative_to(self.project_root)} 以查找最新模型...")
        
        all_models = list(log_root.rglob("model_*.pt"))
        if not all_models:
            print("[信息] 未找到任何模型文件。")
            return
            
        latest_file = max(all_models, key=os.path.getmtime)
        
        latest_path_str = str(latest_file)
        inferred_log_dir = None
        for log_dir_name in self.REVERSE_LOG_MAP.keys():
            if f"{os.path.sep}{log_dir_name}{os.path.sep}" in latest_path_str:
                inferred_log_dir = log_dir_name
                break
        
        if inferred_log_dir:
            self.latest_checkpoint_info = (latest_file, inferred_log_dir)
            print(f"已找到最新模型: {latest_file.relative_to(self.project_root)}")
        else:
            print(f"[警告] 找到了最新模型，但无法从路径推断其日志目录: {latest_file}")


    # --- 核心菜单函数 ---

    @staticmethod
    def _get_model_num(path: Path) -> int:
        """从 model_XXXX.pt 文件名中提取数字 XXXX。"""
        match = re.search(r'model_(\d+)\.pt', path.name)
        return int(match.group(1)) if match else -1

    def _get_display_name(self, option: str | Path) -> str:
        """辅助函数：获取选项的显示名称。"""
        if isinstance(option, Path) and option.name == 'motion.npz':
            return option.parent.name
        elif isinstance(option, Path):
            return option.name
        else:
            return str(option)

    def _interactive_menu(self, 
                          title: str, 
                          options: list[str | Path], 
                          display_strings: list[str] = None
                         ) -> str | Path:
        """使用 simple-term-menu 显示交互式菜单。"""
        
        if not options:
            print("\n" + "=" * 60)
            print(f"=== {title} ===")
            print("=" * 60)
            print("错误：未找到任何选项。")
            if '日期' in title or '时间' in title:
                log_dir_display = self.logs_base_dir.relative_to(self.project_root) if self.logs_base_dir else "'logs/rsl_rl' 目录"
                print(f"请检查日志目录 '{log_dir_display}' 是否存在且包含训练记录。")
            elif '动作文件' in title:
                print(f"请检查 'artifacts' 目录中是否有动作文件。")
            else:
                print("请检查相关目录结构是否正确。")
            sys.exit(1)

        if display_strings is None:
            display_options = [self._get_display_name(opt) for opt in options]
        else:
            if len(display_strings) != len(options):
                print("内部错误：选项和显示字符串的数量不匹配。")
                sys.exit(1)
            display_options = display_strings
        
        terminal_menu = self.TerminalMenu(
            menu_entries=display_options,
            title=title,
            menu_cursor="> ",
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("bg_cyan", "fg_black"),
            cycle_cursor=True,
            status_bar=lambda s: s, 
            status_bar_style=("bg_black", "fg_gray"),
        )
        
        menu_entry_index = terminal_menu.show()
        
        if menu_entry_index is None:
            print("选择已取消。正在退出。")
            sys.exit(0)
        
        selected_option = options[menu_entry_index]
        print(f"您选择了: {display_options[menu_entry_index]}")
        return selected_option

    # --- 流程步骤 ---

    def _select_task(self, title: str, task_list: list = None):
        """第 1 步：选择任务"""
        if task_list is None:
            task_list = self.TASKS
        self.selected_task = self._interactive_menu(title, task_list)

    def _select_motion_file(self):
        """第 2 步：选择动作文件 (仅显示最新版本)"""
        if not self.require_motion_file:
            print("\n已跳过：选择动作文件 (根据配置)。")
            self.selected_motion_file = None
            return

        artifacts_dir = self.project_root / "artifacts"
        print(f"\n正在搜索动作文件于: {artifacts_dir.relative_to(self.project_root)}")
        all_motion_paths = list(artifacts_dir.glob("**/motion.npz"))

        latest_motions = {}
        version_regex = re.compile(r'^(.*?):v(\d+)$')

        for path in all_motion_paths:
            dir_name = path.parent.name
            match = version_regex.match(dir_name)
            base_name = dir_name
            version = -1
            
            if match:
                base_name = match.group(1)
                version = int(match.group(2))
                
            if base_name not in latest_motions or version > latest_motions[base_name][0]:
                latest_motions[base_name] = (version, path)
                
        motion_files = sorted(
            [path for version, path in latest_motions.values()], 
            key=lambda p: p.parent.name
        )
        
        self.selected_motion_file = self._interactive_menu(
            "选择一个动作文件 (MOTION FILE) (仅显示最新版本)", 
            motion_files
        )

    def _select_checkpoint_path_manual(self):
        """(手动流程) 第 3, 4, 5 步：选择日期、时间和模型。"""
        
        # --- 步骤 3: 确定和选择日期 (带预览) ---
        log_dir_name = self.TASK_LOG_MAP.get(self.selected_task, "g1_flat")
        self.logs_base_dir = self.project_root / "logs" / "rsl_rl" / log_dir_name

        if not self.logs_base_dir.exists():
            print(f"错误：日志目录不存在: {self.logs_base_dir}")
            print("请检查您的 TASK_LOG_MAP 是否配置正确，或者是否已开始训练。")
            sys.exit(1)
            
        print(f"\n正在搜索日志目录: {self.logs_base_dir.relative_to(self.project_root)}")

        all_run_dirs = [d for d in self.logs_base_dir.iterdir() if d.is_dir()]
        dir_regex = re.compile(r'^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})(?:_(.*))?$')
        date_to_run_names = {} 

        for d in all_run_dirs:
            match = dir_regex.match(d.name)
            if match:
                date_str = match.group(1)
                run_name = match.group(3) if match.group(3) is not None else "[无运行名称]"
                if date_str not in date_to_run_names:
                    date_to_run_names[date_str] = set()
                date_to_run_names[date_str].add(run_name)
        
        options_list = sorted(list(date_to_run_names.keys()), reverse=True)
        display_list = []
        for date in options_list:
            run_names_str = ", ".join(sorted(date_to_run_names[date]))
            display_list.append(f"{date} (包含: {run_names_str})")

        selected_date = self._interactive_menu(
            "第 3 步 (手动)：选择日期 (DATE) (带内容预览)", 
            options_list,
            display_strings=display_list
        )

        # --- 步骤 4: 选择时间/运行 ---
        time_dirs = [d for d in all_run_dirs if d.name.startswith(str(selected_date))]
        sorted_time_dirs = sorted(time_dirs, key=lambda p: p.name, reverse=True)
        selected_run_dir = self._interactive_menu("第 4 步 (手动)：选择时间/运行 (TIME/RUN)", sorted_time_dirs)

        # --- (新) 步骤 5: 选择模型 (自动或手动) ---
        checkpoint_paths = list(selected_run_dir.glob("model_*.pt"))
        if not checkpoint_paths:
            print(f"错误：在目录 '{selected_run_dir.name}' 中未找到任何 'model_*.pt' 文件。")
            sys.exit(1)
        
        sorted_checkpoints = sorted(checkpoint_paths, key=self._get_model_num, reverse=True)
        latest_model = sorted_checkpoints[0]
        
        # 定义此步骤的选项
        MANUAL_SELECT_ID = "MANUAL_SELECT" # 用一个唯一的ID来标识手动选择
        options_list_step5 = [
            latest_model, # 选项1: 路径对象
            MANUAL_SELECT_ID # 选项2: 字符串ID
        ]
        display_list_step5 = [
            f"[ 自动选择最新 ] {latest_model.name}",
            f"[ 手动选择 ] ...从 {len(sorted_checkpoints)} 个历史模型中选择"
        ]
        
        choice = self._interactive_menu(
            "第 5 步 (手动)：选择模型",
            options_list_step5,
            display_strings=display_list_step5
        )

        if choice == MANUAL_SELECT_ID:
            # 如果用户选择“手动”，则显示所有模型的列表
            self.selected_checkpoint_file = self._interactive_menu(
                f"第 5b 步：选择一个历史模型 (在 {selected_run_dir.name} 中)",
                sorted_checkpoints
            )
        else:
            # 否则，用户选择了 "自动选择最新" (即 latest_model 路径对象)
            self.selected_checkpoint_file = latest_model
            # (选择信息已在 _interactive_menu 中打印)
        
    def _select_num_envs(self):
        """最后一步：输入环境数量。"""
        print("\n" + "=" * 60)
        print(f"=== 最后一步：输入环境数量 (NUM ENVS) ===")
        print("=" * 60)
        while True:
            try:
                num_str = input(f"请输入环境数量 (默认为 {self.default_num_envs})：")
                if not num_str:
                    self.num_envs = self.default_num_envs
                    break
                
                num_int = int(num_str)
                if num_int > 0:
                    self.num_envs = num_int
                    break
                else:
                    print("请输入一个大于0的整数。")
            except ValueError:
                print("无效的输入，请输入一个数字。")
            except KeyboardInterrupt:
                print("\n选择已取消。正在退出。")
                sys.exit(0)
        print(f"您选择了: {self.num_envs} 个环境")

    def run_selection_flow(self):
        """运行完整的交互式选择流程，包含 "快速启动" 选项。"""
        
        # --- 第 0 步：选择启动模式 ---
        options = ["[ 手动选择 ] 手动选择任务、日期..."]
        display_strings = ["[ 手动选择 ] 手动选择任务、日期..."]
        fast_start_id = "FAST_START"
        
        if self.latest_checkpoint_info:
            path, log_dir = self.latest_checkpoint_info
            display = f"[ 快速启动 ] 加载: {path.relative_to(self.project_root)}"
            options.insert(0, fast_start_id)
            display_strings.insert(0, display)
        
        choice = self._interactive_menu(
            "第 0 步：选择启动模式", 
            options, 
            display_strings=display_strings
        )
        
        # --- 根据模式选择后续流程 ---
        if choice == fast_start_id:
            # --- 快速启动流程 ---
            path, log_dir = self.latest_checkpoint_info
            self.selected_checkpoint_file = path
            
            possible_tasks = self.REVERSE_LOG_MAP.get(log_dir, self.TASKS)
            
            self._select_task(
                title="第 1 步 (快速启动)：请确认任务", 
                task_list=possible_tasks
            )
            self._select_motion_file()
            self._select_num_envs()

        else:
            # --- 手动选择流程 ---
            self._select_task(title="第 1 步 (手动)：选择一个任务 (TASK)")
            self._select_motion_file()
            self._select_checkpoint_path_manual() # (已更新为包含步骤 5)
            self._select_num_envs()
        
        # --- 结束，打印总结 ---
        print("\n" + "=" * 60)
        print("=== 正在启动模拟器... ===")
        print(f"  任务 (Task): {self.selected_task}")
        if self.selected_motion_file:
            print(f"  动作 (Motion): {self.selected_motion_file.relative_to(self.project_root)}")
        print(f"  模型 (Model): {self.selected_checkpoint_file.relative_to(self.project_root)}")
        print(f"  环境数 (Num Envs): {self.num_envs}")
        print("=" * 60 + "\n")

if __name__ == "__main__":
    # 这个文件现在是一个模块，不应该被直接运行。
    # 但为了方便测试，您可以取消注释以下代码来单独测试选择器类。
    
    print("--- [测试模式] 正在运行 SmartPlaySelector ---")
    
    selector = SmartPlaySelector(require_motion_file=True)
    selector.run_selection_flow()
    
    print("\n--- [测试模式] 选择结果 ---")
    print(f"Task: {selector.selected_task}")
    print(f"Motion File: {selector.selected_motion_file}")
    print(f"Checkpoint File: {selector.selected_checkpoint_file}")
    print(f"Num Envs: {selector.num_envs}")