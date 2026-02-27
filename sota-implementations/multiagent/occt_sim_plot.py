import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd  # 新增：用于创建DataFrame表格
import matplotlib.font_manager as fm

# --- 论文绘图样式配置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.size'] = 12
# plt.rcParams['axes.labelsize'] = 12
# plt.rcParams['axes.titlesize'] = 12
# plt.rcParams['legend.fontsize'] = 8
# plt.rcParams['xtick.labelsize'] = 9
# plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['axes.grid'] = False

plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
plt.rcParams["figure.dpi"] = 300            # 高清分辨率，确保缩小后仍清晰
plt.rcParams["legend.frameon"] = False      # 去除图例边框，节省空间
font_path = '/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf'
# 定义中文字体属性（宋体，适配小尺寸论文插图）
font_prop_chinese = fm.FontProperties(fname=font_path, size=10)  # 小尺寸适配

# 新增：设置pandas打印格式，避免科学计数法，保留4位小数
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


def extract_rollout_data_simple(file_path, batch_idx=0):
    """从rollout文件中提取所需的速度和位置数据"""
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过该项对比。")
        return None
    
    # 加载数据
    rollouts = torch.load(file_path, map_location='cpu', weights_only=False)
    info = rollouts["next"]["agents"]["info"]
    
    # 确定有效时间步长度
    r_batch = list(rollouts.unbind(0))[batch_idx]
    next_done = r_batch.get(("next", "done")).sum(
        tuple(range(r_batch.batch_dims, r_batch.get(("next", "done")).ndim)),
        dtype=torch.bool,
    )
    valid_len = next_done.nonzero(as_tuple=True)[0][0] + 1 if next_done.any() else rollouts.batch_size[1]
    
    # 提取关键信息
    data = {
        "vel_vec": info["vel"][batch_idx, :valid_len].squeeze(-1).cpu().numpy(),  # [T, N, 2]
        "vel_magnitude": info["vel_norm"][batch_idx, :valid_len].squeeze(-1).cpu().numpy(),  # [T, N]
        "pos": info["pos"][batch_idx, :valid_len].cpu().numpy(),                             # [T, N, 2]
        "ref_vel": info["ref_vel"][batch_idx, :valid_len].squeeze(-1).cpu().numpy(),          # [T, N]
    }
    return data

def calculate_cargo_metrics_and_print(all_data, dt, save_dir, batch_idx):
    """
    计算每个方法大件/前车/后车的运动指标，打印对比表格并保存为CSV
    :param all_data: 字典，key=方法名，value=提取的数据集
    :param dt: 时间步长 (s)
    :param save_dir: 表格保存路径
    :param batch_idx: 批次索引
    :return: 指标DataFrame
    """
    # 严格按要求定义列名，顺序不可变
    col_names = [
        "大件的最大角加速度 (rad/s²)",
        "最大角速度 (rad/s)",
        "大件最大jerk (m/s³)",
        "大件最大acc (m/s²)",
        "大件最大速度 (m/s)",
        "前车最大jerk (m/s³)",
        "前车最大acc (m/s²)",
        "前车最大速度 (m/s)",
        "后车最大jerk (m/s³)",
        "后车最大acc (m/s²)",
        "后车最大速度 (m/s)"
    ]
    # 初始化指标字典，每列对应空列表
    metrics_dict = {col: [] for col in col_names}
    method_names = []  # 存储有效方法名

    # 遍历每个方法计算所有指标
    for method, data in all_data.items():
        method_names.append(method)
        #######################################
        # 1. 计算【大件角运动指标】（复用原代码图4的角速度计算逻辑）
        #######################################
        p_f = data["pos"][:, 0, :]  # 首车位置 [T,2]
        p_r = data["pos"][:, -1, :] # 尾车位置 [T,2]
        delta_p = p_f - p_r
        # 计算航向角并处理跳变（与原代码完全一致）
        heading = np.unwrap(np.arctan2(delta_p[:, 1], delta_p[:, 0]))
        # 大件角速度：航向角一阶差分/dt，补齐一帧（与原代码一致）
        ang_vel = np.diff(heading) / dt
        ang_vel = np.append(ang_vel, ang_vel[-1]) if len(ang_vel) > 0 else np.array([0.0])
        # 大件最大角速度：取绝对值最大值（工程关注幅值）
        max_ang_vel = np.max(np.abs(ang_vel))
        # 大件角加速度：角速度一阶差分/dt，取绝对值最大值
        ang_acc = np.diff(ang_vel) / dt if len(ang_vel) > 0 else np.array([0.0])
        max_ang_acc = np.max(np.abs(ang_acc)) if len(ang_acc) > 0 else 0.0

        #######################################
        # 2. 计算【大件线运动指标】（复用原代码图3的中心速度逻辑）
        #######################################
        v_f_vec = data["vel_vec"][:, 0, :]  # 首车速度矢量 [T,2]
        v_r_vec = data["vel_vec"][:, -1, :] # 尾车速度矢量 [T,2]
        cargo_vel = np.linalg.norm((v_f_vec + v_r_vec) / 2.0, axis=1)  # 大件中心速度 [T]
        # 大件acc/jerk/最大速度
        cargo_acc = np.diff(cargo_vel) / dt if len(cargo_vel) > 0 else np.array([0.0])
        cargo_jerk = np.diff(cargo_acc) / dt if len(cargo_acc) > 0 else np.array([0.0])
        max_cargo_jerk = np.max(np.abs(cargo_jerk)) if len(cargo_jerk) > 0 else 0.0
        max_cargo_acc = np.max(np.abs(cargo_acc)) if len(cargo_acc) > 0 else 0.0
        max_cargo_vel = np.max(cargo_vel) if len(cargo_vel) > 0 else 0.0

        #######################################
        # 3. 计算【前车（首车，索引0）线运动指标】
        #######################################
        front_vel = data["vel_magnitude"][:, 0]  # 前车速度幅值 [T]（与原代码图1一致）
        front_acc = np.diff(front_vel) / dt if len(front_vel) > 0 else np.array([0.0])
        front_jerk = np.diff(front_acc) / dt if len(front_acc) > 0 else np.array([0.0])
        max_front_jerk = np.max(np.abs(front_jerk)) if len(front_jerk) > 0 else 0.0
        max_front_acc = np.max(np.abs(front_acc)) if len(front_acc) > 0 else 0.0
        max_front_vel = np.max(front_vel) if len(front_vel) > 0 else 0.0

        #######################################
        # 4. 计算【后车（尾车，索引-1）线运动指标】
        #######################################
        rear_vel = data["vel_magnitude"][:, -1]  # 后车速度幅值 [T]（与原代码图2一致）
        rear_acc = np.diff(rear_vel) / dt if len(rear_vel) > 0 else np.array([0.0])
        rear_jerk = np.diff(rear_acc) / dt if len(rear_acc) > 0 else np.array([0.0])
        max_rear_jerk = np.max(np.abs(rear_jerk)) if len(rear_jerk) > 0 else 0.0
        max_rear_acc = np.max(np.abs(rear_acc)) if len(rear_acc) > 0 else 0.0
        max_rear_vel = np.max(rear_vel) if len(rear_vel) > 0 else 0.0

        #######################################
        # 5. 将所有指标按列名顺序存入字典
        #######################################
        metrics_dict["大件的最大角加速度 (rad/s²)"].append(max_ang_acc)
        metrics_dict["最大角速度 (rad/s)"].append(max_ang_vel)
        metrics_dict["大件最大jerk (m/s³)"].append(max_cargo_jerk)
        metrics_dict["大件最大acc (m/s²)"].append(max_cargo_acc)
        metrics_dict["大件最大速度 (m/s)"].append(max_cargo_vel)
        metrics_dict["前车最大jerk (m/s³)"].append(max_front_jerk)
        metrics_dict["前车最大acc (m/s²)"].append(max_front_acc)
        metrics_dict["前车最大速度 (m/s)"].append(max_front_vel)
        metrics_dict["后车最大jerk (m/s³)"].append(max_rear_jerk)
        metrics_dict["后车最大acc (m/s²)"].append(max_rear_acc)
        metrics_dict["后车最大速度 (m/s)"].append(max_rear_vel)

    # 创建DataFrame，方法名为索引，严格按要求列序
    metrics_df = pd.DataFrame(metrics_dict, index=method_names)
    
    # 打印美化后的指标对比表格
    print("="*120)
    print(f"批次 {batch_idx} - 大件/前车/后车 运动指标对比表")
    print("="*120)
    print(metrics_df)
    print("="*120)

    # 保存CSV（utf-8-sig避免Excel中文乱码）
    csv_save_path = os.path.join(save_dir, f"batch_idx_{batch_idx}_all_agent_metrics.csv")
    metrics_df.to_csv(csv_save_path, encoding="utf-8-sig")
    print(f"全指标表格已保存为CSV文件：{csv_save_path}\n")

    return metrics_df
def plot_comparison():
    dt = 0.05
    batch_idx = 1
    figsize = (4, 3)
    dir_path = "/home/yons/Graduation/rl_occt/outputs/occt_simulation/"
    # 文件对应的方法标签
    file_map = {
        "rollout_ours.pt": "本文动力学方法",
        "rollout_use_front.pt": "基于前车运动学",
        "rollout_use_rear.pt": "基于后车运动学",
        "rollout_kinematic.pt": "基于加权运动学",
    }
    
    # 加载所有数据
    all_data = {}
    for filename, label in file_map.items():
        data = extract_rollout_data_simple(dir_path+filename, batch_idx=batch_idx)
        if data is not None:
            all_data[label] = data

    if not all_data:
        print("没有加载到有效数据，请检查文件路径。")
        return

    # ---------------------- 新增：调用指标计算和表格打印函数 ----------------------
    calculate_cargo_metrics_and_print(all_data, dt, dir_path, batch_idx)
    # -----------------------------------------------------------------------------

    # 提取参考速度 (假设各文件一致，取第一个)
    first_key = list(all_data.keys())[0]
    ref_vel = all_data[first_key]["ref_vel"][:, 0]
    time = np.arange(len(ref_vel)) * dt
    
    # 定义绘图颜色和线型
    methods = ["本文动力学方法", "基于前车运动学", "基于后车运动学", "基于加权运动学"]
    colors = ["red", "blue", "green", "orange"]
    
    # --- 图 1：第一辆车速度曲线 ---
    plt.figure(figsize=figsize)
    plt.plot(time, ref_vel, 'k--', label='参考速度', alpha=0.7)
    for i, m in enumerate(methods):
        if m in all_data:
            v = all_data[m]["vel_magnitude"][:, 0]
            plt.plot(time[:len(v)], v, color=colors[i], label=m)
    plt.xlabel('时间 (s)', fontproperties=font_prop_chinese)
    plt.ylabel('首车速度 (m/s)', fontproperties=font_prop_chinese)
    plt.legend(prop=font_prop_chinese, frameon=False, loc='upper right')
    plt.tight_layout()
    plt.savefig(dir_path+f'batch_idx_{batch_idx}_front_speed.pdf')
    
    # --- 图 2：最后一辆车速度曲线 ---
    plt.figure(figsize=figsize)
    plt.plot(time, ref_vel, 'k--', label='参考速度', alpha=0.7)
    for i, m in enumerate(methods):
        if m in all_data:
            v = all_data[m]["vel_magnitude"][:, -1]
            plt.plot(time[:len(v)], v, color=colors[i], label=m)
    plt.xlabel('时间 (s)', fontproperties=font_prop_chinese)
    plt.ylabel('尾车速度 (m/s)', fontproperties=font_prop_chinese)
    plt.legend(prop=font_prop_chinese, frameon=False)
    plt.tight_layout()
    plt.savefig(dir_path+f'batch_idx_{batch_idx}_rear_speed.pdf')

    # --- 图 3：超大件中心速度曲线 ---
    plt.figure(figsize=figsize)
    for i, m in enumerate(methods):
        if m in all_data:
            v_f = all_data[m]["vel_vec"][:, 0, :]
            v_r = all_data[m]["vel_vec"][:, -1, :]
            # 矢量均值的模
            v_cargo = np.linalg.norm((v_f + v_r) / 2.0, axis=1)
            plt.plot(time[:len(v_cargo)], v_cargo, color=colors[i], label=m)
    plt.xlabel('时间 (s)', fontproperties=font_prop_chinese)
    plt.ylabel('大件速度 (m/s)', fontproperties=font_prop_chinese)
    plt.legend(prop=font_prop_chinese, frameon=False)
    plt.tight_layout()
    plt.savefig(dir_path+f'batch_idx_{batch_idx}_cargo_center_speed.pdf')

    # --- 图 4：超大件角速度曲线 ---
    plt.figure(figsize=figsize)
    for i, m in enumerate(methods):
        if m in all_data:
            p_f = all_data[m]["pos"][:, 0, :]
            p_r = all_data[m]["pos"][:, -1, :]
            delta_p = p_f - p_r
            # 计算航向角并展开处理跳变
            heading = np.unwrap(np.arctan2(delta_p[:, 1], delta_p[:, 0]))
            ang_vel = np.diff(heading) / dt
            # 补齐一帧
            ang_vel = np.append(ang_vel, ang_vel[-1])
            plt.plot(time[:len(ang_vel)], ang_vel, color=colors[i], label=m)
    plt.xlabel('时间 (s)', fontproperties=font_prop_chinese)
    plt.ylabel('大件角速度 (rad/s)', fontproperties=font_prop_chinese)
    plt.legend(prop=font_prop_chinese, frameon=False)
    plt.tight_layout()
    plt.savefig(dir_path+f'batch_idx_{batch_idx}_cargo_angular_velocity.pdf')

    print("分析图表已成功保存为 PDF 文件。")


if __name__ == "__main__":
    plot_comparison()