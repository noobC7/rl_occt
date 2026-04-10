# Failure Curriculum Bank 方案设计

## 1. 背景与目标

当前 OCCT 任务里，动作突变问题已经通过将 `hinge_status` 替换为连续的 hinge 距边界距离得到缓解，因此后续优化重点不再是动作平滑，而是：

- 提高在困难铰接场景中的恢复能力
- 降低碰撞类失败的重复发生概率
- 让训练资源更多聚焦于“当前策略还不会处理的失败边界”

本方案选择先不修改 actor 结构，只做基于失败样本的 curriculum。

方案核心目标：

- 在不改变任务定义的前提下，提高 hard case 的训练频次
- 不再依赖 `arc_pos` 人为把车辆推到铰接附近
- 使用真实 rollout 中产生的失败案例构建课程样本池
- 当某个失败案例被当前策略成功解决后，将其从课程样本池中删除
- 当某个失败案例被 replay 后仍失败时，用“新的失败前 K 秒状态”替换旧样本，使样本始终跟随当前失败边界演化


## 2. 方案概述

本方案采用两层结构：

1. `Scenario` 侧只负责维护短时状态历史，并在失败时导出“失败前 K 秒”的状态快照。
2. `Trainer` 侧维护长期 `FailureCurriculumBank`，负责跨 epoch 存储、采样、删除和更新失败样本。

整体流程如下：

1. 前 `warmup_epochs` 个 epoch 正常训练，不使用 replay curriculum。
2. 在正常训练过程中，如果某个 episode 因碰撞失败，则从环境中导出“失败前 K 秒”的快照，加入 `FailureCurriculumBank`。
3. warmup 结束后，环境 reset 时以一定概率从 `FailureCurriculumBank` 中抽取一个 snapshot 进行恢复，而不是走默认初始化。
4. 如果 replay 出来的案例在新的 episode 中成功，则从 bank 中删除该样本。
5. 如果 replay 出来的案例在新的 episode 中仍失败，则用本次“新的失败前 K 秒快照”覆盖旧样本。


## 3. 为什么不直接用当前 reset 机制

根据当前代码，OCCT 的 reset 主路径仍然是基于弧长初始化：

- `is_rand_arc_pos`
- `init_arc_pos`
- `init_vel_std`

初始化逻辑位于：

- `../VMAS_occt/vmas/scenarios/occt_scenario.py`

当前 `reset_world_at()` 会根据 `init_arc_pos` 和道路弧长关系重新构造整队的初始位置、姿态和速度，而不是从任意中间状态恢复。

这意味着：

- 当前代码不支持“从碰撞前 K 秒直接 reset”
- 当前只能通过 `arc_pos` 这一类初始化参数间接控制难度

而由于当前场景的初始位置基本都相同，使用 `t=0` 状态没有课程价值，因此需要改成从真实失败 episode 的中间状态恢复。


## 4. 为什么长期 bank 不应放在环境里

长期失败样本池不应放在 `scenario` 内部，原因如下：

- 环境对象更适合维护短时 rollout 状态，不适合跨 epoch 管理长期课程样本
- 若长期 bank 放在环境里，容量通常固定，且容易退化成简单 FIFO，导致有价值旧样本被无差别覆盖
- 训练器更容易做 checkpoint、调度、删除、更新、统计等课程控制逻辑
- 该 bank 不属于 PPO 的训练 replay buffer，它不是给梯度更新采 minibatch 用的，而是给 reset curriculum 用的

因此推荐结构是：

- `Scenario`：短时 `replay_snapshot_buffer`
- `Trainer`：长期 `FailureCurriculumBank`


## 5. 失败案例的定义

当前课程只收集以下三类失败：

- `collision_with_agents`
- `collision_with_lanelets`
- `collision_with_exit_segments`

不收集：

- `goal_incomplete_hinge`
- `hinge_time`

理由：

- 当前讨论中已明确，这两者不作为独立课程来源
- 当前 curriculum 的目标是优先处理会直接导致 episode 失败终止的碰撞类场景


## 6. 为什么需要长度为 `K_steps + 1` 的短时状态缓冲

目标是在失败发生时取到 `t_fail - K_steps` 时刻的状态。

虽然从接口角度看，我们只需要一个“当前减去 K 步”的 snapshot，但要让这个 snapshot 在每个时间步都能正确更新，内部必须维护最近 `K_steps + 1` 个时间步的最小状态历史。

原因是：

- 在当前时刻 `t`，要拿到的是 `state[t - K]`
- 到下一时刻 `t + 1`，要拿到的是 `state[t + 1 - K]`
- 如果内部只保存一个状态值，则无法在下一步自动推得新的 `t - K + 1` 对应状态

因此：

- `margin` 不需要反映在 buffer 长度里
- 但 `size = 1` 不够
- 最小正确设计是 `K_steps + 1`


## 7. `margin` 的作用

`margin` 不是为了决定 buffer 长度，而是为了过滤低价值样本。

含义是：

- 如果失败发生得过早，那么 `t_fail - K_steps` 对应的状态仍然会非常接近统一初始状态
- 这类样本进入 bank 的课程价值不高

因此加入规则：

- 只有当 `t_fail >= K_steps + margin_steps` 时，才允许将该失败样本写入 bank

例子：

- `dt = 0.05`
- `pre_failure_seconds = 1.5`
- 则 `K_steps = 30`
- 若 `margin_steps = 10`
- 则失败必须发生在第 `40` 步及以后，才会收录样本


## 8. 方案的数据流

### 8.1 Warmup 阶段

- 前 `warmup_epochs` 个 epoch：
  - 正常 reset
  - 不使用 curriculum replay
  - 只收集失败样本

### 8.2 收集阶段

每当某个 env 在当前 episode 中出现失败：

1. 判断失败类型是否属于三类碰撞失败
2. 判断当前步数是否满足 `t_fail >= K_steps + margin_steps`
3. 若满足，则从 `replay_snapshot_buffer` 中取出 `t_fail - K_steps` 对应状态
4. 构造 snapshot
5. 交给 trainer 侧 `FailureCurriculumBank` 进行插入或更新

### 8.3 Replay 阶段

warmup 结束后，在环境 reset 时：

1. 以概率 `p_replay` 从 `FailureCurriculumBank` 中抽样
2. 如果命中，则通过 snapshot 恢复该 env 的物理状态
3. 如果未命中，则走默认 reset

### 8.4 Replay 结果处理

对于从 bank 中取出的 replay 样本：

- 如果本轮成功：
  - 从 `FailureCurriculumBank` 删除该 entry

- 如果本轮仍失败：
  - 删除旧 entry
  - 用本轮“新的失败前 K 秒 snapshot”重新插入

这样可以保证 bank 始终保存“当前策略仍未学会”的失败边界。


## 9. `FailureCurriculumBank` 设计

### 9.1 职责

`FailureCurriculumBank` 由 trainer 持有，职责包括：

- 存储长期失败课程样本
- replay 时按概率采样
- 记录样本是否被成功解决
- 成功后删除
- 失败后更新

### 9.2 每条 entry 建议字段

- `entry_id`
- `snapshot`
- `road_id`
- `failure_type`
- `created_epoch`
- `last_used_epoch`
- `times_sampled`
- `times_failed_after_replay`
- `times_succeeded_after_replay`

说明：

- 本方案当前不做按失败类型分桶
- 但仍保留 `failure_type` 元数据，方便后续分析与扩展

### 9.3 Bank 容量

第一版建议：

- `bank_capacity = 4096`

原因：

- 容量足够覆盖当前并行环境下产生的难例
- 不至于积累过多陈旧样本
- 配合“成功删除、失败覆盖更新”机制，bank 会保持相对干净

后续可选调整：

- 若样本多样性不足，可升到 `8192`
- 若长期填不满，可降到 `2048`

### 9.4 替换策略

第一版建议：

- `replace_strategy = oldest_unused`

即：

- bank 未满时直接插入
- bank 已满时替换最久未被采样使用的 entry

当前先不做：

- 按失败类型分桶
- reservoir sampling
- 复杂优先级重加权

### 9.6 当前实际采样约束

当前实现里，`FailureCurriculumBank.sample(...)` 支持按 `road_id` 过滤采样。

原因：

- 当前 batch 内的 `road_id` 槽位是固定重复的，类似 `0,1,2,3,4,5,0,1,2,3,4,5,...`
- 因此 replay reset 不再尝试动态切换 map/path 上下文
- trainer 只让某个 env 槽位从“相同 road_id”的失败样本里采样

这样做的好处：

- 显著降低 replay reset 的实现复杂度
- 避免“车辆状态恢复了，但道路上下文不匹配”的问题
- 当前版本中无需额外修改 `OcctCRMap.reset_splines()` 的道路分配逻辑

### 9.5 去重策略

第一版建议启用轻量去重：

- 若 `road_id` 相同
- 且关键 agent 的位置与朝向差异很小
- 则认为属于同一失败 case
- 直接覆盖旧样本，而不是新增

这样可避免 bank 被大量几乎相同的失败样本占满。


## 10. `Scenario` 侧拟实现内容

### 10.1 新增轻量 `replay_snapshot_buffer`

该 buffer 用于维护最近 `K_steps + 1` 步的最小可恢复状态。

建议存储字段：

- 每个 agent 的 `pos`
- 每个 agent 的 `rot`
- 每个 agent 的 `vel`
- 每个 agent 的 `ang_vel`
- 每个 agent dynamics 的 `cur_delta`
- `agent_s`
- 必要的道路/路径上下文

说明：

- 不存 observation history
- 不存 PPO 训练数据
- 仅存 replay reset 所需最小状态

### 10.2 当前实际实现接口

当前实现里，scenario 侧使用的是一组更贴近 trainer 调度的接口：

1. `configure_failure_curriculum(bank, collect_enabled, enabled, replay_probability, min_bank_size, iteration)`

功能：

- 由 trainer 在每轮调用
- 下发当前 curriculum bank 句柄、是否允许收集失败样本、是否允许 replay、当前 replay 概率、最小 bank 门槛以及当前迭代号

2. `drain_failure_curriculum_events()`

功能：

- trainer 从 scenario 取出本轮累计的课程事件
- 事件类型包括：
  - `new_failure`
  - `replay_success`
  - `replay_failure`

3. `failure_replay_snapshot_buffer`

功能：

- scenario 内部轻量滚动缓冲
- 长度为 `K_steps + 1`
- 用于在失败发生时回看 `t_fail - K_steps` 的可恢复状态

4. `_restore_failure_replay_snapshot(env_index, snapshot, entry_id, agents)`

功能：

- scenario 内部恢复指定 env 的 replay snapshot
- 恢复后直接刷新参考路径、hinge 状态、距离矩阵和碰撞矩阵等派生量

### 10.3 恢复后必须刷新/重置的内容

恢复 snapshot 后，不能只写 `pos/rot/vel`，还必须：

- 恢复 `cur_delta`
- 重新同步 `agent_s`
- 重新计算短期参考路径
- 重新计算 hinge 目标与 hinge 状态
- 重新计算 agent 间距离与 Frenet 距离
- 清空碰撞矩阵
- 重置 reward phase 缓存与相关标志位

恢复策略是：

- 先恢复“物理状态”
- 再复用现有 reset 中的派生量刷新逻辑


## 11. Trainer 侧拟实现内容

### 11.1 新增类

建议在训练脚本侧新增：

- `FailureCurriculumBank`
- 若干 trainer 侧调度辅助函数

其中：

- `FailureCurriculumBank` 负责存储、采样、删除、更新
- trainer 侧辅助函数负责 warmup、采样概率调度以及与 env 通信

### 11.2 Env 与 Bank 的连接

每个 env episode 需要携带两个额外标记：

- `episode_replay_source`
  - `0`：普通 episode
  - `1`：replay episode

- `episode_replay_entry_id`
  - 普通 episode 记为 `-1`
  - replay episode 记为对应 `entry_id`

这样在 episode 结束时，trainer 可以判断：

- 普通 episode 失败：新增样本
- replay episode 成功：删除样本
- replay episode 失败：覆盖旧样本

当前实现中，这两个字段已经通过 scenario `info` 暴露出来：

- `episode_replay_source`
- `episode_replay_entry_id`

### 11.3 Replay 调度

建议：

- `epoch < warmup_epochs`：只收集，不 replay
- `epoch >= warmup_epochs`：以概率 `p_replay` 使用 bank 样本 reset

采样概率建议渐增：

- `replay_prob_start = 0.1`
- `replay_prob_end = 0.3`

不建议第一版超过 `0.3`，避免训练分布偏移过大。


## 12. YAML 配置接口

建议新增如下配置：

```yaml
curriculum:
  enabled: true
  mode: failure_precollision_replay

  warmup_epochs: 50

  replay_prob_start: 0.1
  replay_prob_end: 0.3
  replay_prob_ramp_epochs: 80

  pre_failure_seconds: 1.5
  min_failure_seconds: 2.0
  margin_steps: 10

  bank_capacity: 4096
  min_bank_size: 128

  replace_strategy: oldest_unused

  dedup_enabled: true
  dedup_pos_thresh: 2.0
  dedup_rot_thresh_deg: 10.0

  collect_collision_with_agents: true
  collect_collision_with_lanelets: true
  collect_collision_with_exit_segments: true
```

环境侧补充：

```yaml
env:
  scenario:
    enable_failure_replay_restore: true
    failure_replay_pre_failure_seconds: 1.5
    failure_replay_margin_steps: 10
    n_steps_before_recording: 40
```

说明：

- 若 `dt = 0.05`，`pre_failure_seconds = 1.5`，则 `K_steps = 30`
- `n_steps_before_recording` 需要至少覆盖 `K_steps + 1`
- 实际建议预留到 `40`
- 当前实现里：
  - `failure_replay_pre_failure_seconds` 用于 scenario 内部计算 `failure_replay_k_steps`
  - `failure_replay_margin_steps` 用于决定失败是否足够晚，值得写入 bank

trainer 当前实际读取的核心字段包括：

```yaml
curriculum:
  replay_prob_start: 0.1
  replay_prob_end: 0.3
  replay_prob_ramp_epochs: 80
  bank_capacity: 4096
  min_bank_size: 128
  replace_strategy: oldest_unused
```


## 13. 预期效果

该方案的预期效果不是简单地“增加失败样本数量”，而是让课程样本池逐渐收敛为：

- 当前策略尚未掌握的难例
- 且这些难例对应真实 rollout 中出现的失败边界

预期收益包括：

- 在不改 actor 结构的前提下，提高 hard case 训练密度
- 强化策略在接近碰撞边界时的恢复能力
- 由于成功样本会被移除，课程样本池不会长期保留已解决问题
- 由于失败样本会被新边界覆盖，课程样本会持续跟踪“当前策略最脆弱的位置”


## 14. 当前阶段不做的内容

本方案当前明确不做：

- phase-gated actor
- 失败类型分桶采样
- reservoir sampling
- 按类型优先级重加权
- 使用 PPO 训练 replay buffer 直接承载 curriculum bank
- 基于 `arc_pos` 的人工铰接区初始化 curriculum


## 15. 推荐的最小实现顺序

建议按以下顺序落地：

1. 在 `Scenario` 中实现最小 `replay_snapshot_buffer`
2. 实现 `export_failure_replay_snapshot()` 和 `restore_failure_replay_snapshot()`
3. 在 trainer 中实现 `FailureCurriculumBank`
4. 接入 warmup 与 replay 概率调度
5. 接入 replay episode 的成功删除 / 失败覆盖逻辑
6. 最后再做去重与替换策略完善


## 16. 一句话总结

该方案的本质是：

- 用 `Scenario` 提供“失败前 K 秒”的真实可恢复状态
- 用 `Trainer` 维护一个只保留“当前仍未被解决的失败边界”的课程样本池
- 成功即删除，失败即更新

这比静态 hard-case 采样更适合当前 OCCT 任务，也更符合课程学习的动态目标。


## 17. 当前实验结果（extend_roundabout）

本节记录当前基于以下对比表的实验结论：

- `outputs/occt_comparision/extend_roundabout/comparison_overall.csv`
- `outputs/occt_comparision/extend_roundabout/comparison_roundabout.csv`
- `outputs/occt_comparision/extend_roundabout/comparison_right_angle_turn.csv`
- `outputs/occt_comparision/extend_roundabout/comparison_s_curve.csv`

对比对象为：

- 基线：`ippo_mlp`
- 本方法：`ippo_failure_curriculum_eval`

说明：

- 当前结果基于修正后的评估逻辑
- `all_hinged + exit_segments` 不再计为碰撞
- `success_opportunity_count` 已按固定有效机会数重新定义
- 当前总有效机会数为 `26`

### 17.1 Overall 对比

相对 `ippo_mlp`，`ippo_failure_curriculum_eval` 在 overall 上的主要变化为：

- `success_rate`：`25 / 26 = 0.9615` 提升到 `26 / 26 = 1.0000`
- `success_event_count`：`25 -> 26`
- `collision_rate`：两者均为 `0.0`
- `terminal_success_rate`：两者均为 `1.0`

质量与安全相关指标：

- `s_error_mean`：`0.2684 -> 0.2430`，下降 `9.46%`
- `la_error_mean`：`0.1297 -> 0.1261`，下降 `2.82%`
- `ttc_min_mean`：`10.20 -> 10.66`，提升 `4.45%`
- `hinge_spe_diff_mean`：`0.1621 -> 0.1077`，下降 `33.58%`
- `hinge_ang_diff_mean`：`1.6096 -> 1.4606`，下降 `9.26%`
- `energy_proxy_mean`：`4.0058 -> 3.7900`，下降 `5.39%`

代价与退化项：

- `ste_rate_mean`：`3.3117 -> 3.5212`，上升 `6.33%`
- `hinge_time_mean`：`4.9839 -> 5.1845`，增加 `4.02%`
- `jerk_mean`：`0.3199 -> 0.3210`，基本持平，略差 `0.35%`

总体判断：

- failure curriculum 明显改善了编队误差、铰接速度差、铰接角差和 TTC
- 但没有把铰接效率一起做上去
- 当前方法更像“质量增强器”，而不是“效率增强器”

### 17.2 Roundabout 对比

这是当前 failure curriculum 收益最明显的场景。

相对基线：

- `success_rate`：`11 / 12 = 0.9167` 提升到 `12 / 12 = 1.0000`
- `success_event_count`：`11 -> 12`
- `s_error_mean`：下降 `9.14%`
- `la_error_mean`：下降 `5.01%`
- `jerk_mean`：下降 `2.80%`
- `hinge_spe_diff_mean`：下降 `28.58%`
- `energy_proxy_mean`：下降 `4.49%`

同时也出现：

- `hinge_time_mean`：增加 `6.17%`
- `ste_rate_mean`：增加 `4.77%`

解释：

- failure curriculum 成功补掉了 roundabout 中的最后一个失败机会
- 但策略变得更谨慎，因此完成铰接的时间变长

### 17.3 Right-angle turn 对比

该场景的成功率没有变化，两者均为：

- `success_rate = 8 / 8 = 1.0`

但本方法在质量上仍有改善：

- `s_error_mean`：下降 `8.08%`
- `la_error_mean`：下降 `2.39%`
- `hinge_spe_diff_mean`：下降 `36.76%`
- `hinge_ang_diff_mean`：下降 `10.08%`
- `hinge_time_mean`：下降 `4.24%`

代价：

- `jerk_mean`：上升 `6.32%`
- `ste_rate_mean`：上升 `4.53%`

解释：

- 这是当前唯一一个“质量更好且铰接更快”的场景
- 但控制更激进一些，体现为 jerk 和 steering rate 上升

### 17.4 S-curve 对比

在修正后的有效机会数定义下，S-curve 两者均为：

- `success_rate = 6 / 6 = 1.0`

因此这里不再是“能否突破成功率”的问题，而是“在同样成功的前提下谁更稳”。

相对基线：

- `s_error_mean`：下降 `10.84%`
- `ttc_min_mean`：提升 `15.90%`
- `hinge_spe_diff_mean`：下降 `38.63%`
- `hinge_ang_diff_mean`：下降 `19.70%`
- `energy_proxy_mean`：下降 `10.97%`

但：

- `hinge_time_mean`：增加 `10.92%`
- `ste_rate_mean`：增加 `7.86%`

解释：

- failure curriculum 让 S-curve 中的铰接姿态和速度匹配明显变好
- 但仍表现为“更稳但更慢”

### 17.5 当前实验结论总结

当前这版 failure curriculum 相比基线 `ippo_mlp`，已经实现了：

- overall 成功事件数从 `25 / 26` 提升到 `26 / 26`
- 在 roundabout 中补掉最后一个失败机会
- 在所有 road type 上普遍改善：
  - 编队纵向误差
  - 铰接速度差
  - 铰接角差
  - 能耗代理
- 在多数场景上提高 TTC，说明安全裕度更高

但它也有稳定代价：

- `hinge_time_mean` 通常变长
- `ste_rate_mean` 通常变高
- jerk 没有明显系统性改善

因此，当前方法最准确的定位是：

- 它提高了 hard case 下的鲁棒性与质量
- 但尚未把“更快完成铰接”作为主要优势做出来


## 18. 由结果反推的下一步方向

基于当前 `extend_roundabout` 结果，failure curriculum 这条线的下一步重点不应再是“证明它是否有效”，而应转向：

- 如何降低 replay 训练后策略的保守性
- 如何在保持当前质量收益的前提下压缩 `hinge_time_mean`
- 如何避免 steering rate 随 curriculum 一起上升

从当前结果推断，下一步最值得探索的不是更强的记忆网络，而是：

- curriculum 样本池更新策略
- replay 概率调度
- replay 样本使用后的训练权重
- 以及与铰接效率直接相关的 reward / objective 对齐


## 19. 训练曲线诊断与待优化证据

本节基于训练过程中新增的 `failure_curriculum/*` 曲线进行诊断，目的不是修改当前实现，而是记录后续值得优化的证据、论点和建议。

本次重点观察的曲线包括：

- `failure_curriculum/bank_size`
- `failure_curriculum/active_replays`
- `failure_curriculum/replay_probability`
- `failure_curriculum/total_added`
- `failure_curriculum/total_removed_success`
- `failure_curriculum/total_updated_failure`

### 19.1 当前曲线中没有明显逻辑错误的部分

从训练曲线看，failure curriculum 主机制是跑通的，主要证据如下：

- `replay_probability` 先为 `0`，在 warmup 结束后开始上升，并最终稳定在配置上限附近。
- `active_replays` 在 warmup 期间基本为 `0`，在 replay 启动后逐步升高，并长期保持在非零水平。
- `total_removed_success` 在 warmup 后持续增长，说明 replay 出来的样本确实在被“学会后移除”。
- `total_updated_failure` 也在增长，说明 replay 后仍失败的 case 确实在被新边界更新。

这些现象说明：

- replay curriculum 不是“只写了日志，没有真正参与训练”
- 成功删除与失败更新两套机制都已生效
- trainer 与 scenario 之间的课程样本回传链路已打通

### 19.2 当前最值得关注的问题：bank 前期被过快打满

从曲线来看，当前 `bank_size` 在 warmup 较早阶段就迅速接近 `bank_capacity`，同时：

- `total_added` 很快升到较高水平
- 后续虽然 `bank_size` 逐步下降，但前期的写入速度明显过快

这一点说明：

- 训练早期失败样本写入频率非常高
- bank 很可能在短时间内吸收了大量相似样本
- 当前 `bank_capacity` 的数量级并不等于“有效样本多样性”

换句话说：

- bank 当前可能“容量够大”
- 但并不一定“覆盖面够广”

这并不说明方法失效，但说明 bank 的构造方式仍较粗糙。

### 19.3 为什么这会成为后续优化重点

当前结果里，failure curriculum 已经证明自己有收益：

- 能提升成功事件数
- 能改善 TTC、速度差、角差、能耗代理与编队误差

因此后续重点不应再放在“是否启用 replay curriculum”，而应转向：

- 当前 replay 样本是否足够多样
- 当前 replay 样本是否过多重复集中在某些失败边界附近
- bank 是否在 warmup 期被低质量、重复 hard case 迅速填满

如果这些问题存在，那么它们会带来以下副作用：

- replay 样本池名义容量很大，但有效覆盖不足
- 策略过多反复训练某一类失败边界
- replay 训练可能更偏向保守修正，而不是扩大泛化收益

### 19.4 曲线给出的核心论点

本次训练曲线支持以下论点：

1. **机制层面是有效的**

- replay 被真正启用
- replay 样本被成功解决后会移除
- replay 失败样本会被更新

2. **bank 管理层面仍有改进空间**

- bank 在 warmup 前期被快速写满
- 写入节奏明显快于“高质量课程样本逐步构建”的理想节奏
- 这强烈暗示 bank 中存在较多相似失败样本

3. **当前方法更像“质量修复器”，而不是“效率提升器”**

结合实验结果和训练曲线：

- curriculum 确实在消化 hard cases
- 但它的作用方向更偏向“降低明显错误”
- 而不是“让策略更快完成铰接”

### 19.5 后续建议（仅记录，不立即实现）

以下建议当前仅作为后续优化方向记录，不代表本轮立即修改代码。

#### 建议 1：加强样本去重

当前最直接的优化方向是：

- 提高 `FailureCurriculumBank` 中重复失败样本的过滤能力
- 减少同一路段、相近位置、相近姿态样本被反复加入

目标是让 bank 的“有效多样性”更接近其名义容量。

#### 建议 2：控制新失败样本写入频率

当前 `total_added` 的增长速度偏快，说明样本写入过于密集。

可以考虑：

- 只按一定采样概率记录新失败
- 或对相邻时间/相近状态失败进行节流

目标是让 bank 更像“精选课程样本池”，而不是“所有失败照单全收”。

#### 建议 3：增加 bank 质量诊断日志

当前已有曲线已足以证明机制有效，但若想进一步分析 bank 质量，建议未来新增：

- `failure_curriculum/total_replaced_capacity`
- `failure_curriculum/replay_success_count`
- `failure_curriculum/replay_failure_count`
- `info/episode_replay_source` 的更直接统计版本

这些日志有助于判断：

- bank 满后究竟替换了多少旧样本
- replay 样本被解决的速度
- 当前 collector 中 replay episode 的实际占比

#### 建议 4：保持 replay 概率调度稳定，不优先改 actor

从当前曲线看，问题不在于 replay 概率调度本身。

因此后续若继续优化，优先级应当是：

- 先优化 bank 管理逻辑
- 再看是否需要继续调整 replay 概率
- 当前不建议因为这些曲线直接切回去改 actor 结构

### 19.6 当前诊断结论

训练曲线支持如下判断：

- 当前 failure curriculum **不是无效实现**
- 当前 failure curriculum **已经在稳定工作**
- 当前最值得优化的不是“是否 replay”，而是“bank 中到底存了多少高价值、低重复的样本”

因此，后续如果还有时间，这条线最值得补强的是：

- bank 多样性
- 新样本写入节制
- replay 样本质量诊断

而不是立刻改动 actor 或推翻当前 curriculum 框架。
