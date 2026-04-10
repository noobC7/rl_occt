# OCCT (超限货物协同运输) 项目文档

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. VMAS_occt 仓库详解](#3-vmas_occt-仓库详解)
  - [3.1 核心场景实现](#31-核心场景实现)
  - [3.2 工具类和数据结构](#32-工具类和数据结构)
  - [3.3 地图系统](#33-地图系统)
  - [3.4 脚本工具](#34-脚本工具)
- [4. rl_occt 仓库详解](#4-rl_occt-仓库详解)
  - [4.1 训练算法实现](#41-训练算法实现)
  - [4.2 核心特性](#42-核心特性)
  - [4.3 分析工具](#43-分析工具)
  - [4.4 可视化工具](#44-可视化工具)
- [5. 核心概念说明](#5-核心概念说明)
- [6. 使用指南](#6-使用指南)
- [7. 文件索引](#7-文件索引)
- [8. 术语对照表](#8-术语对照表)

---

## 1. 项目概述

### 1.1 什么是 OCCT？

**OCCT (Over-sized Cargo Collaborative Transportation)** 是一个**超限货物协同运输**的多智能体强化学习项目，旨在解决多辆卡车协同运输超大货物的复杂控制问题。

### 1.2 两个仓库的关系

本项目的代码分布在两个仓库中：

| 仓库 | 路径 | 主要功能 |
|------|------|---------|
| **VMAS_occt** | `/home/yons/Graduation/VMAS_occt` | VMAS仿真环境实现、场景定义、地图系统 |
| **rl_occt** | `/home/yons/Graduation/rl_occt/sota-implementations/multiagent` | 训练算法、分析工具、可视化工具 |

**VMAS_occt** 提供了基础的仿真环境和场景定义，而 **rl_occt** 基于这些环境实现各种强化学习算法和实验工具。

### 1.3 主要功能

- ✅ **多智能体协同控制**：支持多车队列行驶和铰接机制
- ✅ **多种训练算法**：MAPPO、IPPO、MADDPG、MASAC
- ✅ **灵活的课程学习**：线性初始化和相位自适应权重控制
- ✅ **完整的分析工具**：观测空间分析、铰接状态转换分析
- ✅ **丰富的可视化**：仿真轨迹可视化、实验曲线绘制
- ✅ **灵活的地图系统**：支持自定义道路和 CommonRoad 数据

---

## 2. 系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        OCCT 系统架构                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐           ┌──────────────────┐           │
│  │   VMAS_occt      │           │    rl_occt       │           │
│  │   (仿真环境)      │ ────────▶ │   (训练算法)     │           │
│  │                  │           │                  │           │
│  │ • 场景定义        │           │ • MAPPO/IPPO     │           │
│  │ • 地图系统        │           │ • MADDPG/MASAC   │           │
│  │ • 物理引擎        │           │ • 课程学习       │           │
│  └──────────────────┘           └──────────────────┘           │
│           │                              │                      │
│           │ 观测空间                      │ 策略网络             │
│           ▼                              ▼                      │
│  ┌──────────────────┐           ┌──────────────────┐           │
│  │   数据交换层      │           │   分析工具       │           │
│  │                  │           │                  │           │
│  │ • TensorDict     │           │ • 评估指标       │           │
│  │ • 奖励计算       │           │ • 观测分析       │           │
│  │ • 环境重置       │           │ • 可视化         │           │
│  └──────────────────┘           └──────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流向

```
配置文件 (YAML)
    │
    ▼
环境初始化 (VMAS_occt)
    │
    ├─▶ 创建场景 (occt_scenario.py)
    ├─▶ 加载地图 (occt_map.py)
    └─▶ 初始化智能体
    │
    ▼
训练循环 (rl_occt)
    │
    ├─▶ 收集经验 (SyncDataCollector)
    ├─▶ 计算奖励和优势
    ├─▶ 更新策略网络
    └─▶ 记录日志
    │
    ▼
分析和可视化
    │
    ├─▶ 评估指标计算 (occt_metrics_evaluation.py)
    ├─▶ 观测空间分析 (occt_observation_*.py)
    └─▶ 可视化结果 (occt_*_plot.py)
```

---

## 3. VMAS_occt 仓库详解

VMAS_occt 仓库提供了 OCCT 项目的仿真环境和基础设施。

### 3.1 核心场景实现

#### 文件：`occt_scenario.py`

**核心类：`Scenario`**

```python
class Scenario(BaseScenario):
    """
    OCCT 场景的核心实现类，继承自 VMAS 的 BaseScenario
    """
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """
        创建世界和智能体

        Args:
            batch_dim: 批处理维度
            device: 计算设备 (CPU/GPU)
            **kwargs: 其他配置参数

        Returns:
            World: VMAS 世界对象
        """
```

**任务类型枚举：**

| 任务类型 | 值 | 说明 |
|---------|---|------|
| `SIMPLE_PLATOON` | 0 | 简单队列模式（无货物） |
| `OCCT_PLATOON` | 1 | OCCT 模式（带货物铰接） |

**控制方法枚举：**

| 方法类型 | 值 | 说明 |
|---------|---|------|
| `MARL` | 0 | 多智能体强化学习 |
| `PID` | 1 | 传统 PID 控制 |
| `MPPI` | 2 | 模型预测路径积分控制 |

**核心功能：**

1. **世界初始化** (`make_world`)
   - 创建物理世界
   - 初始化智能体
   - 设置地图和参考路径

2. **奖励计算**
   - 进度奖励
   - 速度奖励
   - 车队保持奖励
   - 铰接点约束奖励

3. **观测空间构建**
   - 自身状态（位置、速度、加速度）
   - 参考路径点
   - 铰接点信息（OCCT_PLATOON）
   - 边界距离
   - 其他智能体信息

4. **环境重置**
   - 随机化初始位置
   - 重置物理状态
   - 生成新的参考路径

#### 文件：`occt_scenario_test.py`

用于场景验证和性能评估的测试模块。

---

### 3.2 工具类和数据结构

#### 文件：`occt_utils.py`

提供了一系列用于 OCCT 场景的工具类和数据结构。

**核心工具类：**

##### 1. `OcctConstants` - 预定义常量

```python
class OcctConstants:
    """
    预定义常量类

    属性:
        env_idx_broadcasting: 环境索引广播张量
        empty_action_acc: 空加速度动作
        empty_action_steering: 空转向动作
        mask_pos: 位置掩码
        mask_vel: 速度掩码
        mask_rot: 旋转掩码
        reset_agent_min_distance: 重置时智能体间最小距离
    """
```

##### 2. `OcctNormalizers` - 归一化器

```python
class OcctNormalizers:
    """
    归一化器类，用于将不同尺度的数据标准化

    属性:
        pos: 位置归一化因子
        error_pos: 位置误差归一化因子
        v: 速度归一化因子
        rot: 旋转角度归一化因子
        action_steering: 转向动作归一化因子
        action_acc: 加速度动作归一化因子
        distance_ref: 参考路径距离归一化因子
    """
```

##### 3. `OcctRewards` - 奖励权重

```python
class OcctRewards:
    """
    奖励权重配置

    属性:
        reward_progress: 进度奖励权重
        reward_vel: 速度奖励权重
        reward_goal: 目标奖励权重
        reward_platoon_heading: 车队航向奖励权重
        reward_platoon_space: 车队间距奖励权重
        reward_hinge_space: 铰接点间距奖励权重
        reward_platoon_vel: 车队速度一致性奖励权重
        reward_hinge_vel: 铰接点速度一致性奖励权重
        reward_platoon_ref: 车队参考路径跟踪奖励权重
        reward_hinge_ref: 铰接点参考路径跟踪奖励权重
    """
```

##### 4. `OcctPenalties` - 惩罚项

```python
class OcctPenalties:
    """
    惩罚项配置

    属性:
        near_boundary: 接近边界惩罚
        collision: 碰撞惩罚
        ... 其他惩罚项
    """
```

##### 5. `OcctObservations` - 观察空间定义

```python
class OcctObservations:
    """
    观察空间定义和配置

    属性:
        self_vel: 自身速度
        self_speed: 自身速率
        self_steering: 自身转向角
        self_acc: 自身加速度
        self_ref_velocity: 参考速度
        self_ref_points: 参考路径点
        self_hinge_velocity: 铰接点速度
        self_hinge_points: 铰接点位置
        self_left_boundary_distance: 左边界距离
        self_right_boundary_distance: 右边界距离
        self_hinge_status: 铰接状态
    """
```

##### 6. `OcctThresholds` - 阈值配置

定义各种判断阈值，如碰撞距离、边界距离等。

##### 7. `OcctDistances` - 距离计算

提供智能体之间、智能体与参考路径之间的距离计算方法。

**重要函数：**

```python
def check_hinge_points_in_boundary(hinge_points: Tensor, boundary_pts: List) -> bool:
    """
    检查铰接点是否在边界内

    Args:
        hinge_points: 铰接点坐标
        boundary_pts: 边界点列表

    Returns:
        bool: 是否所有铰接点都在边界内
    """

def get_short_term_hinge_path_by_s(s: float, map_obj: MapBase, n_points: int = 4) -> Tensor:
    """
    根据弧长获取短期铰接路径

    Args:
        s: 当前弧长位置
        map_obj: 地图对象
        n_points: 返回的点数

    Returns:
        Tensor: 铰接路径点 (n_points, 2)
    """

def get_short_term_reference_path_by_s(s: float, map_obj: MapBase, n_points: int = 4) -> Tensor:
    """
    根据弧长获取短期参考路径

    Args:
        s: 当前弧长位置
        map_obj: 地图对象
        n_points: 返回的点数

    Returns:
        Tensor: 参考路径点 (n_points, 2)
    """
```

---

### 3.3 地图系统

#### 文件：`occt_map.py`

**核心类：`MapBase` (抽象基类)**

```python
class MapBase:
    """
    地图抽象基类，定义地图接口
    """
    def get_road_center_pts(self) -> Tensor:
        """获取道路中心点"""

    def get_road_left_pts(self) -> Tensor:
        """获取道路左边界点"""

    def get_road_right_pts(self) -> Tensor:
        """获取道路右边界点"""

    def get_pts(self, s: float) -> Tensor:
        """根据弧长获取路径点"""

    def get_ref_v(self, s: float) -> float:
        """获取参考速度"""
```

**核心类：`OcctMap` (基础地图类)**

```python
class OcctMap(MapBase):
    """
    基础地图类，支持从预定义数据加载地图

    Args:
        road_center_pts: 道路中心点
        road_width: 道路宽度
        ref_velocities: 参考速度分布
    """
```

**核心类：`OcctCRMap` (CommonRoad 地图类)**

```python
class OcctCRMap(MapBase):
    """
    CommonRoad 格式地图类，支持从 CommonRoad 数据集加载地图

    Args:
        cr_map_id: CommonRoad 地图 ID
        ... 其他参数
    """
```

**重要函数：**

```python
def smooth_road_centerline(road_pts: Tensor, smoothing_factor: float = 0.5) -> Tensor:
    """
    平滑道路中心线

    Args:
        road_pts: 原始道路点
        smoothing_factor: 平滑因子

    Returns:
        Tensor: 平滑后的道路点
    """
```

#### 文件：`occt_boundary.py`

**核心类：`OcctBoundaryCalculator`**

```python
class OcctBoundaryCalculator:
    """
    道路边界计算器

    功能:
        - 根据道路中心线计算左右边界
        - 处理边界交点
        - 检查方向约束
    """

    def _calculate_boundary_pts(self, center_pts: Tensor, road_width: float) -> Tuple[Tensor, Tensor]:
        """
        计算边界点

        Args:
            center_pts: 道路中心点
            road_width: 道路宽度

        Returns:
            Tuple[Tensor, Tensor]: (左边界点, 右边界点)
        """

    def _find_nearest_segment_intersection(self, pt: Tensor, segments: List) -> Tensor:
        """
        寻找最近线段交点
        """

    def _check_direction(self, direction: Tensor) -> bool:
        """
        检查方向约束
        """
```

---

### 3.4 脚本工具

#### 1. `occt_map_extractor.py` - 地图提取工具

用于从地图数据中提取特定路径。

**主要功能：**
- 加载原始地图数据
- 提取指定的路径
- 保存处理后的数据

#### 2. `occt_map_point_editor.py` - 地图点编辑器

交互式地图点编辑工具。

**主要功能：**
- 可视化显示道路网络
- 通过点击编辑道路顶点
- 实时预览编辑结果
- 数据验证和保存

#### 3. `occt_map_replace.py` - 地图替换工具

用于替换地图数据中的特定路径。

**主要功能：**
- 替换指定索引的路径
- 验证数据完整性
- 保存修改后的地图

---

## 4. rl_occt 仓库详解

rl_occt 仓库包含了所有的训练算法、分析工具和可视化工具。

### 4.1 训练算法实现

#### 1. `mappo_ippo_occt.py` - MAPPO/IPPO 算法

**主要功能：**
- MAPPO (Multi-Agent PPO) 算法实现
- IPPO (Independent PPO) 算法实现
- 支持集中式训练和分布式执行
- 完整的训练和评估流程

**关键常量：**
```python
AGENT_FOCUS_INDEX = 2  # 焦点智能体索引（用于铰接）
```

**关键函数：**
```python
def build_eval_env(cfg: DictConfig) -> TransformedEnv:
    """
    构建评估环境

    Args:
        cfg: 配置对象

    Returns:
        TransformedEnv: 转换后的环境
    """

def infer_agent_advantage_exclude_dims(cfg: DictConfig) -> List[int]:
    """
    推断优势函数计算时需要排除的维度

    用于处理铰接状态等特殊观测
    """

def log_advantage_layout(observation_layout: List[Dict]) -> None:
    """
    记录观测空间布局
    """

def rendering_callback(env: TransformedEnv, data: TensorDict) -> None:
    """
    渲染回调函数，用于可视化
    """

def run_eval_export_chunk(env: TransformedEnv, policy: TensorDictModule, ...) -> Dict:
    """
    运行评估并导出数据
    """
```

#### 2. `mappo_soft_mlp_occt.py` - 软模块化 MLP

**核心类：`SoftModularMultiAgentMLP`**

```python
class SoftModularMultiAgentMLP(nn.Module):
    """
    软模块化多层感知机网络

    特点:
        - 动态专家网络架构
        - 输入自适应的模块选择
        - 支持多智能体共享或独立参数

    Args:
        n_agent_inputs: 智能体输入维度
        n_agent_outputs: 智能体输出维度
        n_agents: 智能体数量
        centralized: 是否集中式
        share_params: 是否共享参数
        num_modules: 模块数量（默认 2）
        module_hidden: 模块隐藏层维度
    """

    def __init__(self, n_agent_inputs, n_agent_outputs, n_agents,
                 centralized, share_params, device=None, depth=None,
                 num_cells=None, activation_class=nn.Tanh,
                 num_modules=DEFAULT_SOFT_NUM_MODULES, module_hidden=None):
        super().__init__()

        # 为每个智能体创建输入投影层
        self.agent_input_proj = nn.ModuleList()
        for _ in range(self.n_agents):
            self.agent_input_proj.append(
                nn.Linear(n_agent_inputs, num_modules)
            )

        # 创建处理模块
        self.modules = nn.ModuleList()
        for i in range(num_modules):
            self.modules.append(
                nn.ModuleDict({
                    "input_norm": nn.LayerNorm(n_agent_inputs),
                    "fc1": nn.Linear(n_agent_inputs, module_hidden or 64),
                    "activation": activation_class(),
                    "fc2": nn.Linear(module_hidden or 64, module_hidden or 64),
                    "output": nn.Linear(module_hidden or 64, 1),
                })
            )

        # 输出投影层
        self.output_proj = nn.Linear(num_modules, n_agent_outputs)
```

**工作原理：**
1. 每个智能体的输入首先通过输入投影层，生成模块选择权重
2. 输入同时被发送到所有处理模块
3. 各模块的输出根据选择权重进行加权组合
4. 最后通过输出投影层得到最终输出

#### 3. `mappo_lipsnet_occt.py` - LipsNet 架构

**主要功能：**
- 实现 LipsNet (Linearly Interpolated Policy Network) 架构
- 支持相位混合和连续控制
- 集成到 MAPPO/IPPO 训练流程

#### 4. `maddpg_iddpg_occt.py` - MADDPG/IDDPG 算法

**主要功能：**
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 实现
- IDDPG (Independent DDPG) 实现
- 适用于连续动作空间

#### 5. `masac_occt.py` - Multi-Agent SAC

**主要功能：**
- MASAC (Multi-Agent Soft Actor-Critic) 实现
- 最大熵强化学习
- 稳定的训练性能

---

### 4.2 核心特性

#### 1. 线性初始化课程学习

**类：`LinearInitArcCurriculum`**

```python
class LinearInitArcCurriculum:
    """
    线性初始化课程学习

    两个阶段:
        1. 初始化阶段：速度从 0 线性增加到目标速度的 30%
        2. 弧形调整阶段：使用余弦函数平滑过渡到目标速度

    Args:
        init_steps: 初始化阶段步数（默认 500）
        arc_steps: 弧形阶段步数（默认 2000）
        target_max_speed: 目标最大速度（默认 15.0）
        target_min_speed: 目标最小速度（默认 5.0）
    """

    def get_current_max_speed(self, current_step: int) -> float:
        """
        根据当前训练步数计算最大速度

        Args:
            current_step: 当前训练步数

        Returns:
            float: 当前最大速度
        """
        if current_step <= self.init_steps:
            # 初始化阶段：线性增长
            progress = current_step / self.init_steps
            target_speed = self.target_min_speed + (
                self.target_max_speed - self.target_min_speed
            ) * 0.3
            return target_speed * progress
        elif current_step <= self.init_steps + self.arc_steps:
            # 弧形阶段：余弦平滑过渡
            progress = (current_step - self.init_steps) / self.arc_steps
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            low_speed = self.target_min_speed + (
                self.target_max_speed - self.target_min_speed
            ) * 0.3
            high_speed = self.target_max_speed
            return low_speed + (high_speed - low_speed) * cosine_factor
        else:
            # 训练阶段：使用目标最大速度
            return self.target_max_speed
```

**速度曲线示意：**
```
速度
 │
 │     ╱─────────────── 目标最大速度
 │    ╱
 │   ╱  弧形过渡 (余弦曲线)
 │  ╱
 │ ╱
 │╱
 └──────────────────────▶ 训练步数
    ↑init ↑init+arc
```

#### 2. 相位自适应权重控制器

**类：`PhaseAdaptiveWeightController`**

```python
class PhaseAdaptiveWeightController:
    """
    相位自适应权重控制器

    根据训练阶段动态调整不同奖励的权重

    支持的相位类型:
        - base: 基础阶段（简单队列）
        - hinge: 铰接阶段（引入货物）
        - platoon: 车队阶段（完整协同）

    Args:
        phase_configs: 相位配置列表，每个配置包含:
            - start_step: 起始步数
            - end_step: 结束步数
            - weights: 权重字典
            - name: 相位名称
    """

    def __init__(self, phase_configs):
        self.phases = []
        for i, phase_config in enumerate(phase_configs):
            phase = {
                "start_step": phase_config.get("start_step", 0),
                "end_step": phase_config.get("end_step", float('inf')),
                "weights": phase_config.get("weights", {}),
                "name": phase_config.get("name", f"phase_{i}")
            }
            self.phases.append(phase)

    def get_current_phase_weights(self, current_step: int) -> Dict[str, float]:
        """
        获取当前步数的相位权重

        Args:
            current_step: 当前训练步数

        Returns:
            Dict[str, float]: 权重字典
        """
        for phase in self.phases:
            if phase["start_step"] <= current_step < phase["end_step"]:
                return phase["weights"]
        return {"base": 1.0, "hinge": 0.0, "platoon": 0.0}

    def get_current_phase_name(self, current_step: int) -> str:
        """
        获取当前相位名称

        Args:
            current_step: 当前训练步数

        Returns:
            str: 相位名称
        """
        for phase in self.phases:
            if phase["start_step"] <= current_step < phase["end_step"]:
                return phase["name"]
        return "unknown"
```

**使用示例：**
```python
phase_configs = [
    {
        "start_step": 0,
        "end_step": 1000,
        "weights": {"base": 1.0, "hinge": 0.0, "platoon": 0.0},
        "name": "基础训练阶段"
    },
    {
        "start_step": 1000,
        "end_step": 3000,
        "weights": {"base": 0.5, "hinge": 0.5, "platoon": 0.0},
        "name": "铰接引入阶段"
    },
    {
        "start_step": 3000,
        "end_step": float('inf'),
        "weights": {"base": 0.3, "hinge": 0.3, "platoon": 0.4},
        "name": "完整协同阶段"
    }
]

controller = PhaseAdaptiveWeightController(phase_configs)
```

#### 3. 相位掩码提取

**函数：**
```python
def extract_phase_mask(td: TensorDict, phase_key: str = "hinge_status") -> Tensor:
    """
    从 TensorDict 中提取相位掩码

    Args:
        td: TensorDict 对象
        phase_key: 相位键名（默认 "hinge_status"）

    Returns:
        Tensor: 布尔掩码张量

    Raises:
        KeyError: 如果找不到指定的相位键
    """
    phase_tensor = _safe_td_get(td, ("agents", "info", phase_key))
    if phase_tensor is None:
        phase_tensor = _safe_td_get(td, ("next", "agents", "info", phase_key))
    if phase_tensor is None:
        raise KeyError(
            f"Could not find {phase_key} in the sampled TensorDict. "
            "Adaptive phase weighting requires a valid phase indicator in env info."
        )
    return _ensure_phase_mask(phase_tensor)
```

---

### 4.3 分析工具

#### 1. `occt_metrics_evaluation.py` - 评估指标计算

**主要功能：**
- 计算轨迹评估指标
- 支持多种地图类型
- 批量处理验证结果

**核心函数：**
```python
def get_valid_length(data: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算有效轨迹长度

    Args:
        data: 轨迹数据数组
        mask: 有效点掩码（可选）

    Returns:
        float: 有效长度
    """

# 常量定义
DEFAULT_DT = 0.05  # 默认时间步长
DEFAULT_FOLLOWERS = [1, 2, 3]  # 默认跟随车辆索引
DEFAULT_SEDAN_MASS_KG = 1500.0  # 默认轿车质量
DEFAULT_SEDAN_WIDTH_M = 1.5  # 默认车辆宽度
DEFAULT_SEDAN_HEIGHT_M = 1.45  # 默认车辆高度
```

**支持的地图类型：**
```python
ROAD_TYPE_BY_ID = {
    0: "roundabout",      # 环岛
    1: "roundabout",
    2: "right_angle_turn", # 直角转弯
    3: "right_angle_turn",
    4: "s_curve",          # S 弯
    5: "s_curve",
}
```

**方法显示名称：**
```python
METHOD_DISPLAY_NAMES = {
    "marl_baseline": "MARL Baseline",
    "marl_full_obse": "MARL Full Obse",
    "marl_lipsnet++": "MARL LipsNet++",
    "marl_mlp_continuous_w_cp_his10": "MLP Continuous Act Penalty (H=10)",
    "marl_lipsnet_continuous_his10": "LipsNet Continue Baseline (H=10)",
    "marl_lipsnet_phase_blending_his10": "LipsNet Phase Blending (H=10)",
    "marl": "MARL",
    "pid": "PID",
    "mppi": "MPPI",
}
```

#### 2. `occt_observation_jump_analysis.py` - 观测跳跃分析

**主要功能：**
- 分析观测空间中的不连续性
- 检测铰接状态转换时的观测跳跃
- 可视化分析结果

**核心功能：**
```python
def _build_flat_observation_layout(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    构建扁平化观测空间布局

    Args:
        cfg: 配置字典

    Returns:
        List[Dict]: 观测布局列表
    """
    scenario_cfg = cfg["env"]["scenario"]
    task_class = int(scenario_cfg.get("task_class", 1))
    mask_ref_v = bool(scenario_cfg.get("mask_ref_v", False))
    n_points_short_term = int(scenario_cfg.get("n_points_short_term", 4))
    n_nearing_agents = int(scenario_cfg.get("n_nearing_agents_observed", 2))

    layout_spec = [
        ("self_vel", 1),
        ("self_speed", 1),
        ("self_steering", 1),
        ("self_acc", 1),
        ("self_ref_velocity", 0 if mask_ref_v else n_points_short_term),
        ("self_ref_points", n_points_short_term * 2),
    ]

    if task_class == 1:  # OCCT_PLATOON
        layout_spec.extend([
            ("self_hinge_velocity", 0 if mask_ref_v else n_points_short_term),
            ("self_hinge_points", n_points_short_term * 2),
            ("self_left_boundary_distance", self_boundary_distance_dim),
            ("self_right_boundary_distance", self_boundary_distance_dim),
            ("self_hinge_status", n_points_short_term),
            ("self_distance_to_ref", 1),
            ("self_distance_to_hinge", 1),
            ("self_error_vel", 2),
            ("self_platoon_error_space", 2),
        ])
    else:  # SIMPLE_PLATOON
        layout_spec.extend([
            ("self_distance_to_ref", 1),
            ("self_error_vel", 2),
            ("self_platoon_error_space", 2),
        ])

    return layout_spec
```

#### 3. `occt_observation_variance_plot.py` - 观测方差分析

**主要功能：**
- 计算观测空间各维度的方差
- 对比不同方法的观测稳定性
- 生成方差对比图表

#### 4. `occt_hinge_transition_action_analysis.py` - 铰接状态转换分析

**主要功能：**
- 分析铰接状态转换时的动作分布
- 检测动作连续性
- 评估铰接控制质量

---

### 4.4 可视化工具

#### 1. `occt_sim_plot.py` - 仿真轨迹可视化

**主要功能：**
- 绘制智能体轨迹
- 显示参考路径
- 标注碰撞和越界点
- 生成动画

**使用示例：**
```python
from occt_sim_plot import plot_simulation_results

# 绘制仿真结果
plot_simulation_results(
    trajectory_data="path/to/trajectory.csv",
    map_data="path/to/map.pkl",
    output_path="output/plot.png"
)
```

#### 2. `occt_swanlab_plot0325.py` - SwanLab 实验可视化

**主要功能：**
- 从 SwanLab 读取实验数据
- 绘制训练曲线
- 对比不同实验
- 生成实验报告

**支持的可视化：**
- 训练奖励曲线
- 评估指标曲线
- 观测方差分布
- 铰接状态统计

---

## 5. 核心概念说明

### 5.1 铰接机制 (Hinge)

**铰接点**是 OCCT 场景中的核心概念，用于连接多辆卡车和超限货物。

**铰接点配置：**
- 每辆卡车（除了最后一辆）都有一个铰接点
- 铰接点位于卡车后部
- 货物通过铰接点连接到车队

**铰接约束：**
- 铰接点必须在道路边界内
- 铰接点之间保持一定距离
- 铰接角度限制在合理范围内

**铰接状态：**
```python
hinge_status = 1  # 铰接激活（有货物）
hinge_status = 0  # 铰接未激活（无货物）
```

### 5.2 队列管理 (Platoon)

**队列**是指多辆卡车按照一定顺序排列，协同行驶。

**队列特点：**
- 前导车辆（Agent 0）负责导航
- 跟随车辆保持与前车的安全距离
- 所有车辆协调速度和转向

**队列保持奖励：**
```python
reward_platoon_space = -|d - d_target|  # 间距奖励
reward_platoon_vel = -|v_i - v_j|        # 速度一致性奖励
reward_platoon_heading = -|θ_i - θ_j|    # 航向一致性奖励
```

### 5.3 观测空间构建

**观测空间包含以下信息：**

| 观测维度 | 说明 | 维度 |
|---------|------|------|
| `self_vel` | 自身速度向量 | 2 |
| `self_speed` | 自身速率 | 1 |
| `self_steering` | 转向角 | 1 |
| `self_acc` | 加速度 | 1 |
| `self_ref_velocity` | 参考速度（N 个点） | N |
| `self_ref_points` | 参考路径点（N 个点） | 2N |
| `self_hinge_velocity` | 铰接点速度 | N |
| `self_hinge_points` | 铰接点位置 | 2N |
| `self_left_boundary_distance` | 左边界距离 | M |
| `self_right_boundary_distance` | 右边界距离 | M |
| `self_hinge_status` | 铰接状态 | N |
| `self_distance_to_ref` | 到参考路径距离 | 1 |
| `self_distance_to_hinge` | 到铰接点距离 | 1 |
| `self_error_vel` | 速度误差 | 2 |
| `self_platoon_error_space` | 车队间距误差 | 2 |

**总维度：** `6 + 6N + 2M + 5` （其中 N=4, M=10 时为 51 维）

### 5.4 奖励系统设计

**奖励组件：**

1. **进度奖励** (`reward_progress`)
   - 鼓励智能体沿参考路径前进
   - 基于弧长增量

2. **速度奖励** (`reward_vel`)
   - 鼓励保持适当速度
   - 惩罚过低或过高速度

3. **目标奖励** (`reward_goal`)
   - 到达目标点的奖励
   - 完成任务的大额奖励

4. **车队保持奖励**
   - `reward_platoon_heading`: 航向一致性
   - `reward_platoon_space`: 间距保持
   - `reward_platoon_vel`: 速度一致性
   - `reward_platoon_ref`: 参考路径跟踪

5. **铰接约束奖励**
   - `reward_hinge_space`: 铰接点间距
   - `reward_hinge_vel`: 铰接点速度
   - `reward_hinge_ref`: 铰接点路径跟踪
   - `reward_hinge`: 综合铰接奖励

6. **惩罚项**
   - 接近边界惩罚
   - 碰撞惩罚
   - 动作平滑度惩罚
   - Jerk（加速度变化率）惩罚

### 5.5 课程学习机制

**课程学习**是指将训练过程分为多个阶段，逐步增加任务难度。

**OCCT 的课程学习策略：**

1. **阶段 1：基础训练**
   - 使用 SIMPLE_PLATOON 任务
   - 低速度配置
   - 重点学习基础控制

2. **阶段 2：铰接引入**
   - 引入 OCCT_PLATOON 任务
   - 逐步增加铰接权重
   - 中等速度配置

3. **阶段 3：完整协同**
   - 完整的 OCCT_PLATOON 任务
   - 高速度配置
   - 所有组件全权重

---

## 6. 使用指南

### 6.1 环境配置

**依赖安装：**
```bash
# 安装 VMAS
cd /path/to/VMAS_occt
pip install -e .

# 安装 rl_occt 依赖
cd /path/to/rl_occt
pip install -r requirements.txt
```

**主要依赖：**
- `torch`
- `vmas`
- `torchrl`
- `hydra-core`
- `omegaconf`
- `matplotlib`
- `numpy`
- `swanlab` (可选，用于实验追踪)

### 6.2 训练流程

**1. 准备配置文件**

配置文件位于 `config/` 目录下，例如：
```yaml
# config/occt_roundabout_extend/ippo_road_extend_failure_curriculum.yaml
env:
  scenario:
    task_class: 1  # OCCT_PLATOON
    n_agents: 4
    max_speed: 15.0
    ...
training:
  algorithm: "ippo"
  learning_rate: 0.0003
  ...
```

**2. 启动训练**
```bash
python mappo_ippo_occt.py --config-name=occt_roundabout_extend/ippo_road_extend_failure_curriculum
```

**3. 监控训练**
- 使用 SwanLab 可视化训练进度
- 检查日志文件
- 观察奖励曲线

**4. 评估模型**
```bash
python occt_metrics_evaluation.py --checkpoint path/to/checkpoint.pt --eval_env roundabout
```

### 6.3 评估和可视化

**运行评估：**
```bash
python occt_metrics_evaluation.py \
    --validation_result_dir /path/to/results \
    --plot_dir /path/to/output
```

**可视化轨迹：**
```bash
python occt_sim_plot.py \
    --trajectory_path /path/to/trajectory.csv \
    --map_path /path/to/map.pkl \
    --output_path /path/to/plot.png
```

**分析观测空间：**
```bash
python occt_observation_jump_analysis.py \
    --config_path /path/to/config.yaml \
    --checkpoint_path /path/to/checkpoint.pt
```

### 6.4 参数配置说明

**关键配置参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `task_class` | 任务类型 (0:简单, 1:OCCT) | 1 |
| `n_agents` | 智能体数量 | 4 |
| `max_speed` | 最大速度 (m/s) | 15.0 |
| `rod_len` | 货物长度 (m) | 10.0 |
| `n_points_short_term` | 短期参考点数 | 4 |
| `n_nearing_agents_observed` | 观测的邻近智能体数 | 2 |
| `mask_ref_v` | 是否屏蔽参考速度 | False |
| `agent_terminal_mask_enabled` | 是否启用智能体终端掩码 | True |

**课程学习参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `curriculum.init_steps` | 初始化阶段步数 | 500 |
| `curriculum.arc_steps` | 弧形阶段步数 | 2000 |
| `curriculum.target_max_speed` | 目标最大速度 | 15.0 |
| `curriculum.target_min_speed` | 目标最小速度 | 5.0 |

**奖励权重参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `reward_progress` | 进度奖励权重 | 1.0 |
| `reward_platoon_space` | 车队间距奖励权重 | 0.5 |
| `reward_hinge_space` | 铰接间距奖励权重 | 1.0 |
| `reward_hinge_ref` | 铰接路径跟踪权重 | 0.8 |

---

## 7. 文件索引

### 7.1 VMAS_occt 仓库文件列表

| 文件路径 | 功能说明 |
|---------|---------|
| `vmas/scenarios/occt_scenario.py` | 核心场景实现 |
| `vmas/scenarios/occt_scenario_test.py` | 场景测试和验证 |
| `vmas/scenarios/occt_utils.py` | 工具类和数据结构 |
| `vmas/scenarios/occt_map.py` | 地图系统 |
| `vmas/scenarios/occt_boundary.py` | 边界计算 |
| `scripts/occt_map_extractor.py` | 地图提取工具 |
| `scripts/occt_map_point_editor.py` | 地图点编辑器 |
| `scripts/occt_map_replace.py` | 地图替换工具 |

### 7.2 rl_occt 仓库文件列表

| 文件路径 | 功能说明 |
|---------|---------|
| `mappo_ippo_occt.py` | MAPPO/IPPO 算法实现 |
| `mappo_soft_mlp_occt.py` | 软模块化 MLP 实现 |
| `mappo_lipsnet_occt.py` | LipsNet 架构实现 |
| `maddpg_iddpg_occt.py` | MADDPG/IDDPG 算法 |
| `masac_occt.py` | Multi-agent SAC 算法 |
| `occt_metrics_evaluation.py` | 评估指标计算 |
| `occt_observation_jump_analysis.py` | 观测跳跃分析 |
| `occt_observation_variance_plot.py` | 观测方差分析 |
| `occt_hinge_transition_action_analysis.py` | 铰接转换分析 |
| `occt_sim_plot.py` | 仿真轨迹可视化 |
| `occt_swanlab_plot0325.py` | SwanLab 实验可视化 |

### 7.3 关键类索引

#### VMAS_occt

| 类名 | 文件 | 功能 |
|------|------|------|
| `Scenario` | occt_scenario.py | 核心场景类 |
| `MethodClass` | occt_scenario.py | 控制方法枚举 |
| `TaskClass` | occt_scenario.py | 任务类型枚举 |
| `OcctConstants` | occt_utils.py | 常量定义 |
| `OcctNormalizers` | occt_utils.py | 归一化器 |
| `OcctRewards` | occt_utils.py | 奖励权重 |
| `OcctPenalties` | occt_utils.py | 惩罚项 |
| `OcctObservations` | occt_utils.py | 观测空间 |
| `OcctThresholds` | occt_utils.py | 阈值配置 |
| `MapBase` | occt_map.py | 地图基类 |
| `OcctMap` | occt_map.py | 基础地图 |
| `OcctCRMap` | occt_map.py | CommonRoad 地图 |
| `OcctBoundaryCalculator` | occt_boundary.py | 边界计算器 |

#### rl_occt

| 类名 | 文件 | 功能 |
|------|------|------|
| `SoftModularMultiAgentMLP` | mappo_soft_mlp_occt.py | 软模块化 MLP |
| `LinearInitArcCurriculum` | mappo_soft_mlp_occt.py | 线性初始化课程学习 |
| `PhaseAdaptiveWeightController` | mappo_soft_mlp_occt.py | 相位自适应权重控制 |

### 7.4 关键函数索引

#### VMAS_occt

| 函数名 | 文件 | 功能 |
|--------|------|------|
| `check_hinge_points_in_boundary` | occt_utils.py | 检查铰接点是否在边界内 |
| `get_short_term_hinge_path_by_s` | occt_utils.py | 获取短期铰接路径 |
| `get_short_term_reference_path_by_s` | occt_utils.py | 获取短期参考路径 |
| `smooth_road_centerline` | occt_map.py | 平滑道路中心线 |

#### rl_occt

| 函数名 | 文件 | 功能 |
|--------|------|------|
| `build_eval_env` | mappo_ippo_occt.py | 构建评估环境 |
| `infer_agent_advantage_exclude_dims` | mappo_ippo_occt.py | 推断优势函数排除维度 |
| `log_advantage_layout` | mappo_ippo_occt.py | 记录观测空间布局 |
| `rendering_callback` | mappo_ippo_occt.py | 渲染回调 |
| `run_eval_export_chunk` | mappo_ippo_occt.py | 运行评估并导出数据 |
| `extract_phase_mask` | mappo_soft_mlp_occt.py | 提取相位掩码 |
| `get_valid_length` | occt_metrics_evaluation.py | 计算有效轨迹长度 |
| `_build_flat_observation_layout` | occt_observation_jump_analysis.py | 构建观测空间布局 |

---

## 8. 术语对照表

| 英文术语 | 中文翻译 | 说明 |
|---------|---------|------|
| **基础概念** |
| Over-sized Cargo Collaborative Transportation | 超限货物协同运输 | 项目全称 |
| OCCT | 超限货物协同运输 | 项目缩写 |
| Hinge | 铰接点/铰链 | 连接卡车和货物的点 |
| Platoon | 车队/队列 | 多卡车的有序排列 |
| Agent | 智能体 | 单个控制实体（卡车） |
| **强化学习** |
| Reinforcement Learning | 强化学习 | RL |
| Multi-Agent Reinforcement Learning | 多智能体强化学习 | MARL |
| MARL | 多智能体强化学习 | MARL |
| Observation Space | 观测空间 | 状态空间 |
| Action Space | 动作空间 | 控制输入空间 |
| Reward | 奖励 | 反馈信号 |
| Policy | 策略 | 状态到动作的映射 |
| Value Function | 价值函数 | 状态的长期价值 |
| Advantage | 优势 | 相对优势 |
| Episode | 回合 | 一次完整运行 |
| Timestep | 时间步 | 单步时间 |
| **算法名称** |
| MAPPO | Multi-Agent PPO | 多智能体 PPO 算法 |
| IPPO | Independent PPO | 独立 PPO 算法 |
| MADDPG | Multi-Agent DDPG | 多智能体 DDPG 算法 |
| IDDPG | Independent DDPG | 独立 DDPG 算法 |
| MASAC | Multi-Agent SAC | 多智能体 SAC 算法 |
| PPO | Proximal Policy Optimization | 近端策略优化 |
| DDPG | Deep Deterministic Policy Gradient | 深度确定性策略梯度 |
| SAC | Soft Actor-Critic | 软演员-评论家 |
| MPPI | Model Predictive Path Integral | 模型预测路径积分 |
| PID | Proportional-Integral-Derivative | 比例-积分-微分 |
| **网络架构** |
| Soft Modular MLP | 软模块化多层感知机 | 动态专家网络 |
| LipsNet | Linearly Interpolated Policy Network | 线性插值策略网络 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| **训练技术** |
| Curriculum Learning | 课程学习 | 分阶段训练 |
| Phase Adaptive Weighting | 相位自适应权重 | 动态权重调整 |
| Advantage Function | 优势函数 | 优势估计 |
| Policy Gradient | 策略梯度 | 策略优化方法 |
| Value Estimation | 价值估计 | 价值函数学习 |
| **环境相关** |
| Scenario | 场景 | 仿真环境设置 |
| Reference Path | 参考路径 | 目标路径 |
| Boundary | 边界 | 道路边界 |
| Road Centerline | 道路中心线 | 道路中心路径 |
| Frenet Frame | Frenet 坐标系 | 纵横坐标系 |
| Arc Length | 弧长 | 沿路径的距离 |
| **控制和状态** |
| Steering Angle | 转向角 | 方向盘角度 |
| Acceleration | 加速度 | 加速度 |
| Velocity | 速度 | 速度向量 |
| Speed | 速率 | 速度大小 |
| Heading | 航向 | 朝向角度 |
| Yaw Rate | 偏航率 | 转向角速度 |
| Jerk | 加速度变化率 | 加速度导数 |
| **评估指标** |
| Lateral Error | 横向误差 | 横向偏差 |
| Longitudinal Error | 纵向误差 | 纵向偏差 |
| Tracking Error | 跟踪误差 | 跟径偏差 |
| Success Rate | 成功率 | 任务完成率 |
| Collision Rate | 碰撞率 | 碰撞概率 |
| **工具和平台** |
| VMAS | Vectorised Multi-Agent Simulator | 向量化多智能体模拟器 |
| TorchRL | TorchRL | PyTorch 强化学习库 |
| Hydra | Hydra | 配置管理框架 |
| SwanLab | SwanLab | 实验追踪平台 |
| CommonRoad | CommonRoad | 道路场景数据集 |

---

## 附录

### A. 常见问题 (FAQ)

**Q1: 如何添加新的地图？**

A: 有几种方法：
1. 使用 `occt_map_point_editor.py` 交互式编辑
2. 从 CommonRoad 数据集导入（使用 `OcctCRMap`）
3. 创建自定义地图数据文件

**Q2: 如何调整课程学习阶段？**

A: 修改配置文件中的 `curriculum` 部分，或创建自定义的 `PhaseAdaptiveWeightController`。

**Q3: 如何处理观测空间的跳跃问题？**

A: 使用 `occt_observation_jump_analysis.py` 分析跳跃，然后考虑：
1. 调整归一化参数
2. 使用软模块化网络
3. 修改观测空间构造

**Q4: 如何可视化训练过程？**

A:
1. 使用 SwanLab 集成（推荐）
2. 使用 TensorBoard
3. 使用自定义的可视化工具

**Q5: 如何调试铰接约束？**

A:
1. 检查 `hinge_status` 观测
2. 使用 `occt_hinge_transition_action_analysis.py` 分析
3. 调整铰接相关奖励权重

### B. 参考资源

- **VMAS 文档**: https://github.com/proroklab/VectorizedMultiAgentSimulator
- **TorchRL 文档**: https://pytorch.org/rl/
- **CommonRoad**: https://commonroad.in.tum.de/
- **SwanLab**: https://swanlab.cn/

### C. 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

---

**文档版本**: 1.0.0
**最后更新**: 2026-04-08
**维护者**: OCCT 项目组

---

*本文档由 Claude Code 自动生成，基于 OCCT 项目的源代码分析。*
