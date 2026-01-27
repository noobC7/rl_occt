# -*- coding: utf-8 -*-
"""
实验指标可视化对比代码
功能：通过SwanLab OpenApi获取occt项目实验数据，绘制Loss与Reward曲线对比图
适用场景：硕士毕业设计中实验结果的量化分析与可视化展示
"""

# 1. 导入所需依赖库
from swanlab import OpenApi
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ===================== 中文论文绘图样式配置（对齐参考代码）=====================
# 指定宋体字体文件路径（参考代码中的路径）
font_path = '/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf'
# 定义中文字体属性（宋体，适配小尺寸论文插图）
font_prop_chinese = fm.FontProperties(fname=font_path, size=7)  # 小尺寸适配
# 字体大小统一配置（适配小尺寸论文插图）
font_size_label = 7     # 坐标轴标签字体大小（适配3x2画布）
font_size_tick = 6      # 刻度字体大小（数字用新罗马）
font_size_legend = 6    # 图例字体大小
font_size_title = 8     # 标题字体大小

# 全局绘图参数（极致适配小尺寸论文插图+中文显示）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
plt.rcParams["figure.dpi"] = 300            # 高清分辨率，确保缩小后仍清晰
plt.rcParams["legend.frameon"] = False      # 去除图例边框，节省空间
plt.rcParams["axes.titlepad"] = 4           # 减小标题与坐标轴间距
plt.rcParams["axes.labelpad"] = 2           # 减小标签与坐标轴间距
plt.rcParams["grid.alpha"] = 0.3            # 网格透明度（参考代码样式）
plt.rcParams["grid.linestyle"] = "--"        # 网格线型（参考代码样式）
plt.rcParams["grid.linewidth"] = 0.5        # 网格线宽（参考代码样式）

def plot_reward_episode(project: str,exp_cuid: str):
    """
    核心功能函数：获取实验数据并绘制对比曲线
    返回值：无（直接生成并显示可视化图像）
    """
    # 2. 初始化SwanLab OpenApi客户端
    swanlab_api = OpenApi()

    # 3. 获取目标项目（occt）的所有实验列表
    project_experiments = swanlab_api.list_experiments(project=project).data
    print(f"成功获取occt项目下{len(project_experiments)}个实验数据")

    # 5. 批量获取两个实验的关键指标数据
    # 选定指标：训练损失（train_learner_loss_objective）、最大奖励值（train_reward_reward_max）
    target_metrics = ["train_reward_episode_reward_mean", "train_reward_episode_reward_max", "train_reward_episode_reward_min"]
    experiment_1_data = swanlab_api.get_metrics(
        exp_id=exp_cuid,
        keys=target_metrics
    ).data

    # 6. 创建子图布局（论文常用小尺寸，对齐参考代码）
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)  # 保持小尺寸

    # 绘制Reward曲线（颜色/线宽优化，符合论文审美）
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[0]],
        label="平均奖励",  # 英文改中文
        linewidth=1.2,     # 轻微加粗均值线，突出核心趋势
        color='#2E86AB',   # 主色调，便于识别
        alpha=0.9
    )
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[1]],
        label="最大奖励",  # 英文改中文
        linewidth=1.0,
        color='#A23B72',   # 最大值线条色
        alpha=0.7
    )
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[2]],
        label="最小奖励",  # 英文改中文
        linewidth=1.0,
        color='#F18F01',   # 最小值线条色
        alpha=0.7
    )

    # 填充Max和Min之间的区域（保留核心功能）
    ax1.fill_between(
        experiment_1_data.index,  # x轴数据
        experiment_1_data[target_metrics[2]],  # 下界
        experiment_1_data[target_metrics[1]],  # 上界
        alpha=0.2,                # 透明度（0-1，越小越透明）
        color='#C73E1D',          # 填充色
        label='奖励区间'          # 图例改中文
    )
    
    # ===================== 论文级样式配置（核心修改）=====================
    # 标题（中文+宋体）
    ax1.set_title("训练奖励变化曲线", fontproperties=font_prop_chinese, fontsize=font_size_title, fontweight="bold")
    # 坐标轴标签（中文+宋体）
    ax1.set_ylabel("奖励值", fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax1.set_xlabel("训练步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    # 图例（中文+宋体，去除边框）
    ax1.legend(loc="best", fontsize=font_size_legend, prop=font_prop_chinese, frameon=False)
    # 网格（参考代码样式）
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    # 刻度（数字用新罗马，缩小间距）
    ax1.tick_params(
        axis="both", 
        labelsize=font_size_tick, 
        pad=1,
        direction='in',          # 刻度向内（参考代码规范）
        top=False, right=False,  # 隐藏上/右刻度（参考代码规范）
        labelfontfamily='Times New Roman'  # 刻度数字用新罗马
    )

    # 保存图像（改为PDF格式，对齐参考代码；最小化留白）
    plt.savefig(
        os.path.join(save_dir, "platoon_reward.pdf"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05  # 最小化图片留白
    )
    plt.show()

def plot_sim_length(project: str,exp_cuid: str):
    """
    核心功能函数：获取实验数据并绘制对比曲线
    返回值：无（直接生成并显示可视化图像）
    """
    # 2. 初始化SwanLab OpenApi客户端
    swanlab_api = OpenApi()

    # 3. 获取目标项目（occt）的所有实验列表
    project_experiments = swanlab_api.list_experiments(project=project).data
    print(f"成功获取occt项目下{len(project_experiments)}个实验数据")

    # 5. 批量获取两个实验的关键指标数据
    # 选定指标：训练损失（train_learner_loss_objective）、最大奖励值（train_reward_reward_max）
    target_metrics = ["train_episode_length_mean"]
    experiment_1_data = swanlab_api.get_metrics(
        exp_id=exp_cuid,
        keys=target_metrics
    ).data

    # 6. 创建子图布局（论文常用小尺寸，对齐参考代码）
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)  # 保持小尺寸

    # 绘制Reward曲线（颜色/线宽优化，符合论文审美）
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[0]],
        label="平均奖励",  # 英文改中文
        linewidth=1.2,     # 轻微加粗均值线，突出核心趋势
        color='#2E86AB',   # 主色调，便于识别
        alpha=0.9
    )
    # ===================== 论文级样式配置（核心修改）=====================
    # 标题（中文+宋体）
    ax1.set_title("累计步数变化曲线", fontproperties=font_prop_chinese, fontsize=font_size_title, fontweight="bold")
    # 坐标轴标签（中文+宋体）
    ax1.set_ylabel("累计步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax1.set_xlabel("训练步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    # 图例（中文+宋体，去除边框）
    ax1.legend(loc="best", fontsize=font_size_legend, prop=font_prop_chinese, frameon=False)
    # 网格（参考代码样式）
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    # 刻度（数字用新罗马，缩小间距）
    ax1.tick_params(
        axis="both", 
        labelsize=font_size_tick, 
        pad=1,
        direction='in',          # 刻度向内（参考代码规范）
        top=False, right=False,  # 隐藏上/右刻度（参考代码规范）
        labelfontfamily='Times New Roman'  # 刻度数字用新罗马
    )

    # 保存图像（改为PDF格式，对齐参考代码；最小化留白）
    plt.savefig(
        os.path.join(save_dir, "platoon_sim_length.pdf"),  # png改pdf（论文常用矢量格式）
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05  # 最小化图片留白
    )
    plt.show()

# 程序入口
if __name__ == "__main__":
    save_dir = "./outputs/data_vis/"
    os.makedirs(save_dir, exist_ok=True)
    plot_reward_episode("MAPPO_PLATOON","4tgf1du5kpk65bqf4kvk2")
    plot_sim_length("MAPPO_PLATOON","4tgf1du5kpk65bqf4kvk2")
