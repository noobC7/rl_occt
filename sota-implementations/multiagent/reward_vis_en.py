# -*- coding: utf-8 -*-
"""
实验指标可视化对比代码
功能：通过SwanLab OpenApi获取occt项目实验数据，绘制Loss与Reward曲线对比图
适用场景：硕士毕业设计中实验结果的量化分析与可视化展示
"""

# 1. 导入所需依赖库
from swanlab import OpenApi
import matplotlib.pyplot as plt
import os

# 设置matplotlib绘图参数，极致适配小尺寸论文插图
plt.rcParams["font.family"] = ["Times New Roman"]  # 支持英文+中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
plt.rcParams["font.size"] = 6  # 大幅降低全局基础字体大小
plt.rcParams["figure.dpi"] = 300  # 保持高清分辨率，确保缩小后仍清晰
plt.rcParams["legend.frameon"] = False  # 去除图例边框，节省空间
plt.rcParams["axes.titlepad"] = 4  # 减小标题与坐标轴间距
plt.rcParams["axes.labelpad"] = 2  # 减小标签与坐标轴间距

def get_experiment_metrics_and_visualize(exp_cuid: str):
    """
    核心功能函数：获取实验数据并绘制对比曲线
    返回值：无（直接生成并显示可视化图像）
    """
    # 2. 初始化SwanLab OpenApi客户端
    swanlab_api = OpenApi()

    # 3. 获取目标项目（occt）的所有实验列表
    project_experiments = swanlab_api.list_experiments(project="MAPPO_PLATOON").data
    print(f"成功获取occt项目下{len(project_experiments)}个实验数据")

    # 5. 批量获取两个实验的关键指标数据
    # 选定指标：训练损失（train_learner_loss_objective）、最大奖励值（train_reward_reward_max）
    target_metrics = ["train_reward_episode_reward_mean", "train_reward_episode_reward_max", "train_reward_episode_reward_min"]
    experiment_1_data = swanlab_api.get_metrics(
        exp_id=exp_cuid,
        keys=target_metrics
    ).data

    # 6. 创建2行1列的子图布局，极致缩小画布尺寸
    fig, ax1 = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)  # 论文常用小尺寸

    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[0]],
        label="Mean Reward",
        linewidth=1.2,  # 轻微加粗均值线，突出核心趋势
        color='#2E86AB',  # 主色调，便于识别
        alpha=0.9
    )
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[1]],
        label="Max Reward",
        linewidth=1.0,
        color='#A23B72',  # 最大值线条色
        alpha=0.7
    )
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data[target_metrics[2]],
        label="Min Reward",
        linewidth=1.0,
        color='#F18F01',  # 最小值线条色
        alpha=0.7
    )

    # 核心新增：填充Max和Min之间的区域，设置透明色
    ax1.fill_between(
        experiment_1_data.index,  # x轴数据
        experiment_1_data[target_metrics[2]],  # 下界
        experiment_1_data[target_metrics[1]],  # 上界
        alpha=0.2,  # 透明度（0-1，越小越透明）
        color='#C73E1D',  # 填充色，可根据需求调整
        label='Reward Range'  # 可选：添加填充区域的图例
    )
    ax1.set_title("Train Reward Comparison", fontsize=8, fontweight="bold")  # 极致缩小标题字体
    ax1.set_ylabel("Reward", fontsize=7)  # 简化标签文字+缩小字体
    ax1.legend(loc="best", fontsize=6, frameon=False)  # 去除图例边框+缩小字体
    ax1.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)  # 细化网格线
    ax1.tick_params(axis="both", labelsize=6, pad=1)  # 极致缩小刻度字体+减小间距

    # 保存图像：进一步优化保存参数，避免留白过多
    plt.savefig(
        os.path.join(save_dir, "occt_reward.png"),
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
    get_experiment_metrics_and_visualize("4tgf1du5kpk65bqf4kvk2")