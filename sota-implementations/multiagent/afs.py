import numpy as np
import matplotlib.pyplot as plt

# 全局参数
STEP_DELTA = 0.05  # 最大更新步长
K = 3.0            # Sigmoid陡峭度
EPS = 1e-6
INIT_WEIGHT = 3.0  # 初始奖励值
TARGET_WEIGHT = 5.0# 目标奖励值
STEPS = 200        # 仿真步数

# Sigmoid平滑仿真（对应最终版代码逻辑）
def sigmoid_smooth_simulation():
    weight = INIT_WEIGHT
    prev_delta = 0.0  # 初始速率为0
    weight_list = [weight]
    delta_list = [prev_delta]
    
    for _ in range(STEPS):
        error = TARGET_WEIGHT - weight
        if np.abs(error) < EPS:
            current_delta = 0.0
        else:
            # 计算理想速率
            ideal_delta = STEP_DELTA * np.sign(error)
            target_delta = np.clip(ideal_delta, -STEP_DELTA, STEP_DELTA)
            # 速率差 + Sigmoid因子
            delta_gap = target_delta - prev_delta
            sigmoid_factor = 1.0 / (1.0 + np.exp(-K * (np.abs(delta_gap)/STEP_DELTA - 0.5)))
            # 计算当前速率
            current_delta = prev_delta + sigmoid_factor * delta_gap
            current_delta = np.clip(current_delta, -STEP_DELTA, STEP_DELTA)
        # 更新权重
        weight += current_delta
        # 保存数据
        weight_list.append(weight)
        delta_list.append(current_delta)
        prev_delta = current_delta
    return np.array(weight_list), np.array(delta_list)

# 一阶平滑仿真（对比组，原代码默认模式）
def first_order_simulation():
    weight = INIT_WEIGHT
    weight_list = [weight]
    delta_list = [0.0]
    for _ in range(STEPS):
        error = TARGET_WEIGHT - weight
        if np.abs(error) < EPS:
            current_delta = 0.0
        else:
            current_delta = STEP_DELTA * np.sign(error)
            current_delta = np.clip(current_delta, -STEP_DELTA, STEP_DELTA)
        weight += current_delta
        weight_list.append(weight)
        delta_list.append(current_delta)
    return np.array(weight_list), np.array(delta_list)

# 运行仿真
sigmoid_weight, sigmoid_delta = sigmoid_smooth_simulation()
first_weight, first_delta = first_order_simulation()
x = np.arange(0, STEPS+1)

# 绘图
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 子图1：奖励权重变化曲线
ax1.axhline(y=TARGET_WEIGHT, color='r', linestyle='--', label='目标值=5.0', linewidth=2)
ax1.plot(x, sigmoid_weight, color='#2E86AB', label='Sigmoid平滑（最终版）', linewidth=2.5)
ax1.plot(x, first_weight, color='#C73E1D', label='一阶平滑（对比）', linestyle='-.', linewidth=2)
ax1.set_ylabel('奖励权重值', fontsize=12, fontweight='bold')
ax1.set_title('奖励值从3→5的变化曲线（Sigmoid vs 一阶平滑）', fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(2.8, 5.2)

# 子图2：更新速率（Δ）变化曲线（核心看0启动）
ax2.plot(x, sigmoid_delta, color='#2E86AB', label='Sigmoid平滑速率', linewidth=2.5)
ax2.plot(x, first_delta, color='#C73E1D', label='一阶平滑速率', linestyle='-.', linewidth=2)
ax2.axhline(y=STEP_DELTA, color='g', linestyle='--', label='最大速率=0.05', linewidth=1.5)
ax2.set_xlabel('迭代步数', fontsize=12, fontweight='bold')
ax2.set_ylabel('权重更新速率 (Δ)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, STEPS)

plt.tight_layout()
plt.savefig('reward_sigmoid_smooth_curve.png', dpi=300, bbox_inches='tight')
plt.show()