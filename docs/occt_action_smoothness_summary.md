# OCCT 动作平滑与观测连续性实验总结

## 1. 主题范围

本文档整理当前围绕 OCCT 多智能体编队/铰接任务所做的动作平滑性相关实验、已确认结论、已经落地的修复，以及当前明确的后续网络方案。文档只保留已经验证过的结论和当前可执行方案，不保留中间反复讨论过程。

本文档对应的核心基线与变体主要包括：

- `MAPPO + MLP + full observation + continuous observation`
- `MAPPO + LipsNet / LipsNet phase blending`
- `remove_hinge_status_from_observation=True` 的 MLP 基线
- reward 平滑切换与更大 steering change penalty 的对照实验

---

## 2. 当前已经确认的核心结论

### 2.1 提高观测连续性和语义稳定性，确实有助于提高动作输出平滑性

这是当前最明确的结论。

- 动作突变的主要来源并不只是策略网络本身，而是 observation 中存在离散切换、数值伪突变或阶段语义突变。
- 当观测中的硬切换和伪突变被压低后，`jerk` 和 `steering rate` 都会改善。
- 相比继续在 reward 上做平滑或一味加大动作变化惩罚，优化观测连续性更有效。

### 2.2 在当前任务里，reward 硬切换不是动作突变的主导原因

通过以下对照实验可得：

- 增大 `penalty_change_steering` 5 倍并没有进一步降低动作突变。
- 将 reward 从硬切换改为 S 型平滑切换，也没有显著降低 `ste_rate_mean`，只对个别行为指标产生轻微影响。

因此当前更合理的判断是：

- reward 切换会影响训练和局部行为；
- 但动作突变的主导来源仍然是 observation/phase cue 的不连续，以及 actor 在切换处学出的不连续控制映射。

### 2.3 LipsNet 在“观测含噪”时有价值，但在当前“观测主要问题不是噪声而是切换/表示不连续”的设定下，会带来动作迟缓风险

当前已经确认：

- LipsNet 的 Fourier filter 和 Lipschitz 正则确实能让动作更平顺；
- 但在当前任务中，观测本身并没有明显随机噪声，主要问题是阶段切换和观测语义不连续；
- 在这种设定下继续使用 LipsNet，会把一些对铰接有用的快速几何变化也一起“磨平”，从而造成：
  - 动作更平滑；
  - 但铰接响应更慢；
  - 有时会带来碰撞和性能下降。

因此当前更合适的主线是：

- 基线优先使用“全连续 observation + MLP”；
- LipsNet 作为针对含噪观测的备选方法，而不是当前主线。

### 2.4 `self_hinge_status` 是切换型动作突变的重要诱因之一，但不是唯一原因

通过 `remove_hinge_status_from_observation=True` 的对照实验可确认：

- 在 road0、road3 等典型切换场景中，原 baseline 在 `hinge_status` 变化附近存在明显 steering 突变；
- 去掉 `self_hinge_status` 后，这类“和 hinge_status 同步发生的局部尖跳”显著减弱，甚至基本消失；
- 说明二值 `hinge_status` 作为 observation 中的硬开关，会直接诱发一部分 phase 切换型动作突变。

但同时也确认：

- 去掉 `self_hinge_status` 以后，全局 worst-case 的 steering jump 并没有完全消失；
- 剩余突变会转移到其他观测量上，尤其是 `self_hinge_info` 和 `others_relative_acceleration`。

因此更准确的结论是：

- `self_hinge_status` 是局部切换型突变的重要诱因；
- 去掉它是有效的；
- 但它不是全局唯一根因。

---

## 3. 关键实验结果

### 3.1 MARL 三种方法对比结论

三类 MARL 方法对比后，当前已确认：

- `MARL-LipsNet-PB` 的平顺性最好；
- `jerk_mean`、`ste_rate_mean` 以及极端控制动作都更小；
- 说明相位融合确实能把动作输出“磨平”，尤其在环岛场景更明显。

但代价也很明确：

- 纵向编队精度下降，`s_error_mean/std` 偏大；
- 铰接效率与铰接质量未同步提升，`hinge_time_mean` 更长、`hinge_ang_diff` 更大；
- 安全裕度略有下降，TTC 偏低；
- 本质上是用一部分编队精度、铰接效率和安全冗余，换取了更好的平顺性。

因此对于当前主线，`MARL-LipsNet-PB` 更像是“平顺性上限参考”，而不是最终最优基线。

### 3.2 增大 `penalty_change_steering` 的实验结论

以 `marl_mlp_continuous_w_cp_his10` 为基线：

- 基线：
  - `jerk_mean = 0.3104`
  - `ste_rate_mean = 2.8685`
  - `hinge_time_mean = 5.9528`
  - `success_rate = 1.0`
  - `collision_rate = 0.0`

- 将 `penalty_change_steering` 增大 5 倍后：
  - `jerk_mean = 0.3118`
  - `ste_rate_mean = 3.5631`
  - `success_rate = 0.8333`
  - `collision_rate = 0.1667`

结论：

- 单纯增大 steering 变化惩罚并没有进一步平滑动作；
- 反而会破坏成功率和碰撞表现；
- 该方向不应继续作为主优化方向。

### 3.3 reward S 型平滑切换实验结论

使用 S 曲线平滑切换 reward 后：

- `jerk_mean = 0.3074`，仅略好于基线；
- `ste_rate_mean = 2.8959`，没有改善；
- `success_rate = 0.8333`
- `collision_rate = 0.1667`

结论：

- reward 平滑切换不是当前动作突变的主解法；
- 它可能改变局部行为和价值学习路径；
- 但并没有显著解决动作输出突变问题。

### 3.4 当前更优的 MLP 基线表现

在修复 `others_rot`、压缩 `others_relative_acceleration` 输入、拆分 `self_error_vel` 后，新的 MLP 基线表现更接近“平顺性与效率兼顾”：

- `jerk_mean` 明显下降；
- `ste_rate_mean` 保持在可接受范围；
- `hinge_time_mean` 未明显恶化，甚至略有改善；
- `success_rate = 1.0`
- `collision_rate = 0.0`

说明当前主方向“优化 observation 连续性 + 保持 MLP actor”是成立的。

---

## 4. 已经定位并确认的问题

### 4.1 `others_rot` 的伪突变来自相对角度没有做 wrap

已确认 bug：

- 相对角度原先直接使用 `rotations_global - rot_i`；
- 未做 `2pi` 主值消解；
- 导致 observation 中 `others_rot` 在跨越角度边界时出现接近 `1.0` 的伪突变。

修复后：

- `others_rot` 的大跳变基本消失；
- 说明该 bug 已被正确定位并修复。

### 4.2 `others_relative_acceleration` 原来将整段 history 直接拼入 observation，会制造窗口滚动伪突变

已确认问题：

- 原来将整段相对加速度历史直接展开输入；
- 相邻时刻 observation 比较时，整个时间窗口会平移；
- 即使底层物理变化连续，向量表示也会产生很大相邻差分。

已做修改：

- 不再输入完整 history；
- 仅保留当前时刻的相对加速度摘要值。

当前状态：

- 该项仍然是剩余突变源之一；
- 但已经明显弱于修复前。

### 4.3 `self_error_vel` 原来根据 `hinge_status` 直接切换，存在语义跳变

原问题：

- 原 observation 中的 `self_error_vel` 在 platoon 和 hinge 两种逻辑之间直接切换；
- 会在阶段切换时引入硬突变。

已做修改：

- 将其拆成两路并行量：
  - `self_platoon_error_vel`
  - `self_hinge_error_vel`
- reward 仍可使用原逻辑；
- observation 不再直接吃这个切换后的变量。

### 4.4 `self_hinge_info` 中真正主导突变的不是位置，而是 boundary signal

对 `self_hinge_info` 的分量拆解后已确认：

- 最容易在 steering jump 时同步大跳的，不是 hinge 点的 `x/y`，也不是速度；
- 而是每个预瞄点中的 `boundary_signal`。

这意味着：

- 当前连续 hinge cue 中，最尖锐、最容易触发切换反应的，是 boundary/availability 这部分；
- 后续如果继续做 observation 优化，应优先处理它。

---

## 5. 当前 observation 归一化范围结论

分析 rollout：

- [rollout_iter_0_frames_0_paths_0_5.pt](/home/yons/Graduation/rl_occt/outputs/occt_comparision/extend_road_comparison/11-10-15_mlp_full_obse_road_extend_more_hinge_reward_offset_eval/run-20260330_111019-252xzzfyqi9szo2rifdfq/rollouts/rollout_iter_0_frames_0_paths_0_5.pt)

### 当前最需要调整归一化因子的组

- `self_right_boundary_distance`
- `self_left_boundary_distance`
- `others_distance`
- `self_hinge_error_vel`

这些组在当前数据上并没有稳定落在 `[-1,1]` 内，说明尺度偏小。

### 当前存在明显饱和的组

- `self_hinge_info`

该组已经频繁打满边界，说明不是简单的“归一化正好”，而是已经存在信息饱和和分辨率损失。

### 当前不是第一优先级的组

- `others_pos`
- `self_error_space`
- `self_distance_to_ref`
- `self_ref_points`
- `others_relative_longitudinal_velocity`

这些组存在少量超界，但主体分布仍可接受。

### 当前整体较健康的组

- `self_vel`
- `self_speed`
- `self_steering`
- `self_acc`
- `self_ref_velocity`
- `self_platoon_error_vel`
- `others_rot`

### 特殊项

- `others_relative_acceleration`

该项不是“整体归一化因子偏小”，而是少量尖峰较大，不建议仅靠整体缩放解决。

---

## 6. 当前已经落地的 observation 侧改动

目前已经实现并验证的 observation 相关改动包括：

- 修复 `others_rot` 的 wrap 问题；
- 将 `others_relative_acceleration` 改成当前值摘要；
- 将 `self_error_vel` 拆成 `self_platoon_error_vel` 和 `self_hinge_error_vel`；
- 支持 `remove_hinge_status_from_observation=True`；
- 增加 action-level 平滑性监控指标：
  - `command_jerk`
  - `command_jerk_abs`
  - `steering_rate_deg`
  - `steering_rate_abs_deg`

---

## 7. 当前阶段的总判断

到目前为止，当前话题可以总结为：

- **动作突变的主因是 observation 的不连续和切换信号的硬开关，而不是 reward 切换本身。**
- **提高 observation 连续性，已经被证明能够显著改善动作平滑性。**
- **LipsNet 在当前无明显噪声的 observation 设定下，不是最合适的主线方法。**
