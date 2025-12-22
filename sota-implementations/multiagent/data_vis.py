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

def get_experiment_metrics_and_visualize(save_dir: str):
    """
    核心功能函数：获取实验数据并绘制对比曲线
    返回值：无（直接生成并显示可视化图像）
    """
    # 2. 初始化SwanLab OpenApi客户端
    swanlab_api = OpenApi()

    # 3. 获取目标项目（occt）的所有实验列表
    project_experiments = swanlab_api.list_experiments(project="occt").data
    print(f"成功获取occt项目下{len(project_experiments)}个实验数据")

    # 4. 选取前两个实验作为对比对象（毕业设计实验对比样本）
    experiment_1_cuid = project_experiments[0].cuid
    experiment_2_cuid = project_experiments[1].cuid
    print(f"待对比实验1 CUID：{experiment_1_cuid}")
    print(f"待对比实验2 CUID：{experiment_2_cuid}")

    # 5. 批量获取两个实验的关键指标数据
    # 选定指标：训练损失（train_learner_loss_objective）、最大奖励值（train_reward_reward_max）
    target_metrics = ["train_learner_loss_objective", "train_reward_reward_max"]
    experiment_1_data = swanlab_api.get_metrics(
        exp_id=experiment_1_cuid,
        keys=target_metrics
    ).data
    experiment_2_data = swanlab_api.get_metrics(
        exp_id=experiment_2_cuid,
        keys=target_metrics
    ).data

    # 6. 创建2行1列的子图布局，极致缩小画布尺寸
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2), constrained_layout=True)  # 论文常用小尺寸

    # 7. 绘制第一个子图：Train Learner Loss对比曲线（极致压缩样式）
    ax1.plot(
        experiment_1_data.index,
        experiment_1_data["train_learner_loss_objective"],
        label="Exp1 Loss",  # 简化图例文字，节省空间
        linewidth=1.0,  # 最小化线条宽度，适配小尺寸
        alpha=0.8
    )
    ax1.plot(
        experiment_2_data.index,
        experiment_2_data["train_learner_loss_objective"],
        label="Exp2 Loss",
        linewidth=1.0,
        alpha=0.8
    )
    ax1.set_title("Train Learner Loss Comparison", fontsize=8, fontweight="bold")  # 极致缩小标题字体
    ax1.set_ylabel("Loss", fontsize=7)  # 简化标签文字+缩小字体
    ax1.legend(loc="best", fontsize=6, frameon=False)  # 去除图例边框+缩小字体
    ax1.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)  # 细化网格线
    ax1.tick_params(axis="both", labelsize=6, pad=1)  # 极致缩小刻度字体+减小间距

    # 8. 绘制第二个子图：Train Reward Max对比曲线（同步极致压缩）
    ax2.plot(
        experiment_1_data.index,
        experiment_1_data["train_reward_reward_max"],
        label="Exp1 Max Reward",
        linewidth=1.0,
        alpha=0.8
    )
    ax2.plot(
        experiment_2_data.index,
        experiment_2_data["train_reward_reward_max"],
        label="Exp2 Max Reward",
        linewidth=1.0,
        alpha=0.8
    )
    ax2.set_title("Train Max Reward Comparison", fontsize=8, fontweight="bold")
    ax2.set_xlabel("Step", fontsize=7)  # 简化标签文字+缩小字体
    ax2.set_ylabel("Max Reward", fontsize=7)
    ax2.legend(loc="best", fontsize=6, frameon=False)
    ax2.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    ax2.tick_params(axis="both", labelsize=6, pad=1)

    # 保存图像：进一步优化保存参数，避免留白过多
    plt.savefig(
        os.path.join(save_dir, "occt_experiment_metrics_comparison.png"),
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
    get_experiment_metrics_and_visualize(save_dir)