# -*- coding: utf-8 -*-
"""
实验指标可视化对比代码
功能：通过SwanLab OpenApi获取occt项目实验数据，绘制Loss与Reward曲线对比图
适用场景：硕士毕业设计中实验结果的量化分析与可视化展示
"""

# 1. 导入所需依赖库
from plt_cn_utils_bigger import *
from swanlab import OpenApi
import os
import numpy as np
import matplotlib.pyplot as plt
CUT_ITER=100


def _set_xaxis_to_valid_range(ax, x_values):
    """将 x 轴限制在有效数据范围内，避免 matplotlib 自动留白。"""
    x_array = np.asarray(x_values)
    if x_array.size == 0:
        ax.set_xlim(0.0, 1.0)
        return

    x_start = 0.0
    x_end = float(x_array[-1])
    if x_end <= x_start:
        x_end = x_start + 1e-9
    ax.set_xlim(x_start, x_end)


def plot_reward_episode(project: str,exp_cuid: str,target_metrics: list, file_name: str, y_label: str, color: str):
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
    
    experiment_1_data = swanlab_api.get_metrics(
        exp_id=exp_cuid,
        keys=target_metrics
    ).data
    x_values = experiment_1_data.index[:CUT_ITER]

    # 6. 创建子图布局（论文常用小尺寸，对齐参考代码）
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)  # 保持小尺寸

    # 绘制Reward曲线（颜色/线宽优化，符合论文审美）
    ax1.plot(
        x_values,
        experiment_1_data[target_metrics[0]][:CUT_ITER],
        label="平均奖励",  # 英文改中文
        linewidth=1.2,     # 轻微加粗均值线，突出核心趋势
        color=color,   # 主色调，便于识别
        alpha=0.9
    )
    ax1.plot(
        x_values,
        experiment_1_data[target_metrics[1]][:CUT_ITER],
        linewidth=0.6,
        color=color,   # 最大值线条色
        alpha=0.4
    )
    ax1.plot(
        x_values,
        experiment_1_data[target_metrics[2]][:CUT_ITER],
        linewidth=0.6,
        color=color,   # 最小值线条色
        alpha=0.4
    )

    # 填充Max和Min之间的区域（保留核心功能）
    ax1.fill_between(
        x_values,  # x轴数据
        experiment_1_data[target_metrics[2]][:CUT_ITER],  # 下界
        experiment_1_data[target_metrics[1]][:CUT_ITER],  # 上界
        alpha=0.2,                # 透明度（0-1，越小越透明）
        color=color,          # 填充色
        label='奖励区间'          # 图例改中文
    )
    
    ax1.set_ylabel(y_label, fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax1.set_xlabel("训练步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    # 图例（中文+宋体，去除边框）
    #ax1.legend(loc="best", fontsize=font_size_legend, prop=font_prop_chinese, frameon=False)
    # 网格（参考代码样式）
    #ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    # 刻度（数字用新罗马，缩小间距）
    ax1.tick_params(
        axis="both", 
        labelsize=font_size_tick, 
        pad=1,
        direction='in',          # 刻度向内（参考代码规范）
        top=False, right=False,  # 隐藏上/右刻度（参考代码规范）
        labelfontfamily='Times New Roman'  # 刻度数字用新罗马
    )
    _set_xaxis_to_valid_range(ax1, x_values)

    # 保存图像（改为PDF格式，对齐参考代码；最小化留白）
    plt.savefig(
        os.path.join(save_dir, file_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05  # 最小化图片留白
    )
    #plt.show()
    plt.close()

def plot_sim_length(project: str,exp_cuid: str, target_metrics: list, file_name: str, color: str):
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
    experiment_1_data = swanlab_api.get_metrics(
        exp_id=exp_cuid,
        keys=target_metrics
    ).data
    x_values = experiment_1_data.index[:CUT_ITER]

    # 6. 创建子图布局（论文常用小尺寸，对齐参考代码）
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)  # 保持小尺寸

    # 绘制Reward曲线（颜色/线宽优化，符合论文审美）
    ax1.plot(
        x_values,
        experiment_1_data[target_metrics[0]][:CUT_ITER],
        label="平均奖励",  # 英文改中文
        linewidth=1.2,     # 轻微加粗均值线，突出核心趋势
        color=color,   # 主色调，便于识别
        alpha=0.9
    )
    step_curve = np.array([experiment_1_data[target_metrics[i]][:CUT_ITER] for i in range(len(target_metrics))])
    upper_bound = np.max(step_curve, axis=0)  
    # 计算下界曲线：每一列（相同位置）的最小值，shape=(500,)
    lower_bound = np.min(step_curve, axis=0)  
    # 填充Max和Min之间的区域（保留核心功能）

    ax1.plot(
        x_values,
        upper_bound,
        linewidth=0.6,
        color=color,   # 最大值线条色
        alpha=0.4
    )
    ax1.plot(
        x_values,
        lower_bound,
        linewidth=0.6,
        color=color,   # 最小值线条色
        alpha=0.4
    )
    ax1.fill_between(
        x_values,  # x轴数据
        lower_bound,  # 下界
        upper_bound,  # 上界
        alpha=0.2,                # 透明度（0-1，越小越透明）
        color=color,          # 填充色
        label='奖励区间'          # 图例改中文
    )
    # 坐标轴标签（中文+宋体）
    ax1.set_ylabel("累计步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax1.set_xlabel("训练步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    # 图例（中文+宋体，去除边框）
    #ax1.legend(loc="best", fontsize=font_size_legend, prop=font_prop_chinese, frameon=False)
    # 网格（参考代码样式）
    #ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    # 刻度（数字用新罗马，缩小间距）
    ax1.tick_params(
        axis="both", 
        labelsize=font_size_tick, 
        pad=1,
        direction='in',          # 刻度向内（参考代码规范）
        top=False, right=False,  # 隐藏上/右刻度（参考代码规范）
        labelfontfamily='Times New Roman'  # 刻度数字用新罗马
    )
    _set_xaxis_to_valid_range(ax1, x_values)

    # 保存图像（改为PDF格式，对齐参考代码；最小化留白）
    plt.savefig(
        os.path.join(save_dir, file_name),  # png改pdf（论文常用矢量格式）
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05  # 最小化图片留白
    )
    #plt.show()
    plt.close()

# 程序入口
if __name__ == "__main__":
    save_dir = "./outputs/data_vis_0411/"
    experiment_id="4a8ziqj2asgxj9xcon37b"
    project_name="MAPPO_PLATOON_0411"
    os.makedirs(save_dir, exist_ok=True)

    target_metrics = ["train_reward_episode_reward_mean", "train_reward_episode_reward_max", "train_reward_episode_reward_min"]
    target_metrics = ["reward/episode_reward_mean", "reward/episode_reward_max", "reward/episode_reward_min"]
    plot_reward_episode(project_name,experiment_id,target_metrics,"train_reward_episode.pdf","训练奖励值", color='blue')
    target_metrics = ["eval/episode_reward_mean", "eval/episode_reward_max", "eval/episode_reward_min"]
    plot_reward_episode(project_name,experiment_id,target_metrics,"eval_reward_episode.pdf","评估奖励值", color='red')
    target_metrics = ["train_reward_reward_mean", "train_reward_reward_max", "train_reward_reward_min"]
    target_metrics = ["reward/reward_mean", "reward/reward_max", "reward/reward_min"]
    plot_reward_episode(project_name,experiment_id,target_metrics,"train_reward_step.pdf","单步训练奖励值", color='orange')
    target_metrics = ["road/road_mean_step"]+["road/road_"+str(i)+"_mean_step" for i in range(2)]
    target_metrics = ["road/road_mean_step","road/road_max_step","road/road_min_step"]
    plot_sim_length(project_name,experiment_id,target_metrics,"sim_length.pdf", color='green')
