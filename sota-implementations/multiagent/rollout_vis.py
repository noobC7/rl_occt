import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import os
import subprocess
import time
import webbrowser
import plotly.io as pio
import pickle
import argparse
from scipy.signal import savgol_filter
pio.templates.default = "plotly_white"
def smooth_data(data, window_length=51, polyorder=3):
    """
    使用 Savitzky-Golay 滤波器对数据进行平滑
    data: 输入数组 [Time, Agents]
    window_length: 窗口长度，必须是奇数。值越大越平滑，但细节损失越多
    polyorder: 多项式拟合阶数。通常取 2 或 3
    """
    # 确保窗口长度小于数据长度，且为奇数
    T = data.shape[0]
    if T <= window_length:
        window_length = T - 1 if (T % 2 == 0) else T # 保证奇数且不超过长度
        if window_length < polyorder + 2:
            return data # 数据太短，不平滑
            
    # 对 axis=0 (时间轴) 进行平滑
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)
def extract_rollout_data(rollouts):
    """从rollout对象中提取所需数据"""
    # 获取基础数据
    batch_size, time_steps, num_agents = rollouts["agents"].batch_size
        # 获取有效时间步（参考logging.py中的实现）
    rollout_list = list(rollouts.unbind(0))  # 按batch维度解绑
    valid_time_steps = []
    
    for batch_idx, r in enumerate(rollout_list):
        # 计算done字段的总和，确定轨迹结束位置
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        
        # 找到第一个done的位置
        if next_done.any():
            done_index = next_done.nonzero(as_tuple=True)[0][0]  # 第一个done索引
            valid_len = done_index + 1  # 有效时间步长度
            valid_time_steps.append(valid_len)
            print(f"Batch {batch_idx}有效时间步：{valid_len}")
        else:
            valid_time_steps.append(time_steps)  # 如果没有done，使用全部时间步
            print(f"Batch {batch_idx}有效时间步：{time_steps} (无done标记)")
    
    # 确保数据在CPU上
    data = {
        "time_step": np.arange(time_steps),
        "agent_id": np.arange(num_agents),
        "batch_id": np.arange(batch_size),
        "valid_time_steps": valid_time_steps  # 新增：有效时间步列表
    }
    
    data["action_log_probs"] = rollouts["agents"]["action_log_prob"].cpu().numpy()  # [batch, time, agent]
    
    info = rollouts["next"]["agents"]["info"]
    data["act_steer"] = info["act_steer"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["act_acc"] = info["act_acc"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["pos"] = info["pos"].cpu().numpy()  # [batch, time, agent, 2]
    data["error_space"] = info["error_space"].cpu().numpy()  # [batch, time, agent, 2]
    data["error_vel"] = info["error_vel"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    if "hinge_dis" in info.keys():
        data["hinge_dis"] = info["hinge_dis"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["hinge_status"] = info["hinge_status"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["agent_hinge_status"] = info["agent_hinge_status"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["reward_track_hinge"] = info["reward_track_hinge"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["reward_track_hinge_vel"] = info["reward_track_hinge_vel"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["reward_hinge"] = info["reward_hinge"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["reward_approach_hinge"] = info["reward_approach_hinge"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["vel_magnitude"] = info["vel_norm"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["ref_vel"] = info["ref_vel"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["rot"] = info["rot"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["distance_ref"] = info["distance_ref"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["distance_lookahead_pts"] = info["distance_lookahead_pts"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["distance_left_b"] = info["distance_left_b"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["distance_right_b"] = info["distance_right_b"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["is_collision_with_agents"] = info["is_collision_with_agents"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["is_collision_with_lanelets"] = info["is_collision_with_lanelets"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_total"] = info["reward_total"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_progress"] = info["reward_progress"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_vel"] = info["reward_vel"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_goal"] = info["reward_goal"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_track_ref_vel"] = info["reward_track_ref_vel"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_track_ref_space"] = info["reward_track_ref_space"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_track_ref_heading"] = info["reward_track_ref_heading"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["reward_track_ref_path"] = info["reward_track_ref_path"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_change_steering"] = info["penalty_change_steering"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_change_acc"] = info["penalty_change_acc"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_collide_with_agents"] = info["penalty_collide_with_agents"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_outside_boundaries"] = info["penalty_outside_boundaries"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_near_boundary"] = info["penalty_near_boundary"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_near_other_agents"] = info["penalty_near_other_agents"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    data["penalty_backward"] = info["penalty_backward"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
    return data, batch_size, time_steps, num_agents
    
    
def plot_agent_data(data, batch_idx=0, agent_idx=0):
    has_hinge = "hinge_dis" in data.keys()
    valid_time_steps = data["valid_time_steps"][batch_idx]
    time_steps = data["time_step"][:valid_time_steps]
    color_list=["blue","purple","green","orange","brown","red","black","cyan","magenta","gray","olive","pink","teal","navy","salmon","turquoise"]
    if agent_idx==0 or agent_idx==3:
        vel_mag = data["vel_magnitude"][batch_idx, :valid_time_steps, agent_idx]
        dt = 0.05
        vel_diff = np.diff(vel_mag)
        acc_diff = vel_diff / dt
        acc_calc = np.concatenate([np.array([0.0]), acc_diff], axis=0)
    else:
        acc_calc = data["act_acc"][batch_idx, :valid_time_steps, agent_idx]
    plot_groups = [
        (
            "Speed / Ref Vel",
            [
                ("Speed", data["vel_magnitude"][batch_idx, :valid_time_steps, agent_idx], color_list[0], "speed"),
                ("Ref Vel", data["ref_vel"][batch_idx, :valid_time_steps, agent_idx], color_list[10], "ref_vel"),
            ],
        ),
        (
            "Vel Error / Acceleration",
            [
                ("Vel Error", data["error_vel"][batch_idx, :valid_time_steps, agent_idx], color_list[5], "vel_error"),
                ("Acceleration", acc_calc, color_list[1], "acceleration"),
            ],
        ),
        (
            "Heading / Steering [rad]",
            [
                ("Heading", data["rot"][batch_idx, :valid_time_steps, agent_idx], color_list[2], "heading_angle"),
                ("Steering", data["act_steer"][batch_idx, :valid_time_steps, agent_idx], color_list[3], "steering_angle"),
            ],
        ),
        (
            "Reference Distances",
            [
                ("Distance to Reference", data["distance_ref"][batch_idx, :valid_time_steps, agent_idx], color_list[4], "distance_ref"),
                ("Distance to Lookahead Pts", data["distance_lookahead_pts"][batch_idx, :valid_time_steps, agent_idx], color_list[12], "distance_lookahead_pts"),
            ],
        ),
        (
            "Boundary Distances",
            [
                ("Distance to Left Boundary", data["distance_left_b"][batch_idx, :valid_time_steps, agent_idx], color_list[1], "distance_left_b"),
                ("Distance to Right Boundary", data["distance_right_b"][batch_idx, :valid_time_steps, agent_idx], color_list[2], "distance_right_b"),
            ],
        ),
        (
            "Space Error",
            [
                ("Space Front", data["error_space"][batch_idx, :valid_time_steps, agent_idx, 0], color_list[6], "space_front"),
                ("Space Back", data["error_space"][batch_idx, :valid_time_steps, agent_idx, 1], color_list[7], "space_back"),
            ],
        ),
        (
            "Reward Total",
            [
                ("Reward Total", data["reward_total"][batch_idx, :valid_time_steps, agent_idx], color_list[0], "reward_total"),
            ],
        ),
        (
            "Reward Progress / Vel",
            [
                ("Reward Progress", data["reward_progress"][batch_idx, :valid_time_steps, agent_idx], color_list[1], "reward_progress"),
                ("Reward Vel", data["reward_vel"][batch_idx, :valid_time_steps, agent_idx], color_list[2], "reward_vel"),
            ],
        ),
        (
            "Reward Goal / Ref Vel",
            [
                ("Reward Goal", data["reward_goal"][batch_idx, :valid_time_steps, agent_idx], color_list[3], "reward_goal"),
                ("Reward Track Ref Vel", data["reward_track_ref_vel"][batch_idx, :valid_time_steps, agent_idx], color_list[4], "reward_track_ref_vel"),
            ],
        ),
        (
            "Reward Ref Space / Heading",
            [
                ("Reward Track Ref Space", data["reward_track_ref_space"][batch_idx, :valid_time_steps, agent_idx], color_list[5], "reward_track_ref_space"),
                ("Reward Track Ref Heading", data["reward_track_ref_heading"][batch_idx, :valid_time_steps, agent_idx], color_list[8], "reward_track_ref_heading"),
            ],
        ),
        (
            "Reward Ref Path",
            [
                ("Reward Track Ref Path", data["reward_track_ref_path"][batch_idx, :valid_time_steps, agent_idx], color_list[9], "reward_track_ref_path"),
            ],
        ),
    ]

    if has_hinge:
        plot_groups.extend(
            [
                (
                    "Reward Hinge Capture",
                    [
                        ("Reward Track Hinge", data["reward_track_hinge"][batch_idx, :valid_time_steps, agent_idx], color_list[10], "reward_track_hinge"),
                        ("Reward Track Hinge Vel", data["reward_track_hinge_vel"][batch_idx, :valid_time_steps, agent_idx], color_list[11], "reward_track_hinge_vel"),
                    ],
                ),
                (
                    "Reward Hinge / Approach",
                    [
                        ("Reward Hinge", data["reward_hinge"][batch_idx, :valid_time_steps, agent_idx], color_list[12], "reward_hinge"),
                        ("Reward Approach Hinge", data["reward_approach_hinge"][batch_idx, :valid_time_steps, agent_idx], color_list[13], "reward_approach_hinge"),
                    ],
                ),
                (
                    "Hinge Distance / Status",
                    [
                        ("Hinge Dis", data["hinge_dis"][batch_idx, :valid_time_steps, agent_idx], color_list[0], "hinge_dis"),
                        ("Hinge Status", data["hinge_status"][batch_idx, :valid_time_steps, agent_idx], color_list[1], "hinge_status"),
                    ],
                ),
                (
                    "Hinge Approach / Diagnostics",
                    [
                        ("Reward Approach Hinge", data["reward_approach_hinge"][batch_idx, :valid_time_steps, agent_idx], color_list[13], "reward_approach_hinge_diag"),
                        ("Agent Hinge Status", data["agent_hinge_status"][batch_idx, :valid_time_steps, agent_idx], color_list[14], "agent_hinge_status"),
                    ],
                ),
            ]
        )

    plot_groups.extend(
        [
            (
                "Penalty Control",
                [
                    ("Penalty Change Steering", data["penalty_change_steering"][batch_idx, :valid_time_steps, agent_idx], color_list[14], "penalty_change_steering"),
                    ("Penalty Change Acc", data["penalty_change_acc"][batch_idx, :valid_time_steps, agent_idx], color_list[15], "penalty_change_acc"),
                ],
            ),
            (
                "Penalty Agent Safety",
                [
                    ("Penalty Near Other Agents", data["penalty_near_other_agents"][batch_idx, :valid_time_steps, agent_idx], color_list[6], "penalty_near_other_agents"),
                    ("Penalty Collide with Agents", data["penalty_collide_with_agents"][batch_idx, :valid_time_steps, agent_idx], color_list[7], "penalty_collide_with_agents"),
                ],
            ),
            (
                "Penalty Boundary Safety",
                [
                    ("Penalty Near Boundary", data["penalty_near_boundary"][batch_idx, :valid_time_steps, agent_idx], color_list[8], "penalty_near_boundary"),
                    ("Penalty Outside Boundaries", data["penalty_outside_boundaries"][batch_idx, :valid_time_steps, agent_idx], color_list[9], "penalty_outside_boundaries"),
                ],
            ),
            (
                "Penalty Backward",
                [
                    ("Penalty Backward", data["penalty_backward"][batch_idx, :valid_time_steps, agent_idx], color_list[5], "penalty_backward"),
                ],
            ),
            (
                "Action Log Prob",
                [
                    ("Action Log Prob", data["action_log_probs"][batch_idx, :valid_time_steps, agent_idx], color_list[10], "action_log_prob"),
                ],
            ),
        ]
    )

    n_cols = 5
    num_rows = (len(plot_groups) + n_cols - 1) // n_cols
    subplot_titles = [title for title, _ in plot_groups]
    subplot_titles.extend([""] * (num_rows * n_cols - len(subplot_titles)))
    fig = make_subplots(
        rows=num_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
    )

    for idx, (_, traces) in enumerate(plot_groups):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        for trace_name, trace_y, color, legendgroup in traces:
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=trace_y,
                    mode='lines',
                    name=trace_name,
                    line=dict(color=color),
                    legendgroup=legendgroup,
                    showlegend=True,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f'Agent {agent_idx} Data Analysis (Batch {batch_idx})',
        height=320 * num_rows,
        width=1800,
        hovermode='x unified'
    )

    return fig
def create_summary_dashboard(data, batch_idx=0):
    """创建汇总仪表板（4个独立图表，每个图表图例在右上方）"""
    # 方案2.1：返回4个独立图表（推荐，灵活性更高）
    figs = []
    
    # 1. 轨迹图
    fig1 = go.Figure()
    t = data["time_step"]  # 时间步数组
    valid_time_steps = data["valid_time_steps"][batch_idx]
    for agent_idx in range(data["pos"].shape[2]):
        positions = data["pos"][batch_idx, :valid_time_steps, agent_idx]
        x_vals = positions[:, 0]
        y_vals = positions[:, 1]
        
        # 为每个点创建包含时间信息的悬停文本
        hover_text = [
            f"Agent {agent_idx}<br>Time Step: {int(step)}<br>X: {x:.2f}<br>Y: {y:.2f}"
            for step, x, y in zip(t, x_vals, y_vals)
        ]
        
        fig1.add_trace(go.Scatter(
            x=x_vals, 
            y=y_vals,
            mode='lines',  # 恢复为只显示线条
            name=f'Agent {agent_idx}',
            line=dict(width=2),  # 移除颜色渐变设置
            text=hover_text,     # 绑定悬停文本
            hoverinfo='text'     # 悬停时只显示自定义文本
        ))

    fig1.update_layout(
        title=f'Agent Trajectories (Batch {batch_idx})',
        height=400,
        width=500,
        hovermode='x unified',  # 恢复原有的悬停模式
        # 图例放在右上方
        legend=dict(
            x=1.0, y=1.0,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'  # 半透明白色背景，避免遮挡
        )
    )

    fig1.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    figs.append(fig1)
    
    # 2. 速度比较
    fig2 = go.Figure()
    for agent_idx in range(data["vel_magnitude"].shape[2]):
        fig2.add_trace(go.Scatter(
            x=data["time_step"][:valid_time_steps],
            y=data["vel_magnitude"][batch_idx, :valid_time_steps, agent_idx],
            mode='lines',
            name=f'Agent {agent_idx}',
            line=dict(width=1.5)
        ))
    fig2.update_layout(
        title=f'Speed Comparison (Batch {batch_idx})',
        height=400,
        width=500,
        hovermode='x unified',
        legend=dict(
            x=1.0, y=1.0,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    figs.append(fig2)
    
    # 3. 航向角比较
    fig3 = go.Figure()
    for agent_idx in range(data["rot"].shape[2]):
        fig3.add_trace(go.Scatter(
            x=data["time_step"][:valid_time_steps],
            y=data["rot"][batch_idx, :valid_time_steps, agent_idx],
            mode='lines',
            name=f'Agent {agent_idx}',
            line=dict(width=1.5)
        ))
    fig3.update_layout(
        title=f'Heading Angle Comparison (Batch {batch_idx})',
        height=400,
        width=500,
        hovermode='x unified',
        legend=dict(
            x=1.0, y=1.0,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    figs.append(fig3)
    
    # 4. 到参考路径的距离
    fig4 = go.Figure()
    for agent_idx in range(data["distance_ref"].shape[2]):
        fig4.add_trace(go.Scatter(
            x=data["time_step"][:valid_time_steps],
            y=data["distance_ref"][batch_idx, :valid_time_steps, agent_idx],
            mode='lines',
            name=f'Agent {agent_idx}',
            line=dict(width=1.5)
        ))
    fig4.update_layout(
        title=f'Distance to Reference (Batch {batch_idx})',
        height=400,
        width=500,
        hovermode='x unified',
        legend=dict(
            x=1.0, y=1.0,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    figs.append(fig4)
    
    return figs

def visualize_rollout(data, num_agents, output_dir="./rollout_visualizations", batch_idx=0, html_file_name="rollout_visualization.html"):
    """主可视化函数 - 简化版，只生成一个包含所有agent仪表板的HTML文件"""
    os.makedirs(output_dir, exist_ok=True)
    html_content = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rollout 可视化汇总</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .chart-container {
                margin-bottom: 40px;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2 {
                color: #333;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <h1>Rollout 可视化汇总</h1>
    '''
    
    # 添加汇总仪表板
    dashboard_fig = create_summary_dashboard(data, batch_idx=batch_idx)
    dashboard_div = [dash.to_html(full_html=False, include_plotlyjs=True) for dash in dashboard_fig]
    
    html_content += f'''
        <div class="chart-container">
            <h2>汇总仪表板</h2>
            <!-- 单栏容器：Flex并排，自动换行适配屏幕 -->
            <div class="single-column-charts" style="display: flex; gap: 15px; flex-wrap: wrap; align-items: flex-start;">
                {''.join(dashboard_div)}
            </div>
        </div>
    '''
        
    
    # 为每个agent添加详细仪表板（2x4布局）
    for agent_idx in range(num_agents):
        agent_fig = plot_agent_data(data, batch_idx=batch_idx, agent_idx=agent_idx)
        agent_div = agent_fig.to_html(full_html=False, include_plotlyjs=False)
        html_content += f'''
        <div class="chart-container">
            <h2>Agent {agent_idx} 详细数据</h2>
            {agent_div}
        </div>
        '''
    
    # 完成HTML内容
    html_content += '''
    </body>
    </html>
    '''
    
    # 保存HTML文件
    output_path = os.path.join(output_dir, html_file_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"可视化结果已保存到: {output_path}")
    
    # 返回主要图表对象
    return {
        "dashboard": dashboard_fig
    }

def load_rollout(rollout_path):
    """
    加载保存的rollout对象
    
    参数:
        rollout_path: rollout文件路径
    
    返回:
        rollouts: 加载的TensorDict对象
    """
    if not os.path.exists(rollout_path):
        raise FileNotFoundError(f"Rollout file not found: {rollout_path}")
    
    # 添加weights_only=False参数以兼容tensordict对象的加载
    rollouts = torch.load(rollout_path, map_location=torch.device('cpu'), weights_only=False)
    
    return rollouts

def visualize_your_rollout(data, num_agents, output_dir="./rollout_visualizations", batch_idx=0, html_file_name="rollout_visualization.html"):
    """
    可视化rollout数据并提供本地网页映射链接
    
    参数:
        rollouts: rollout输出的TensorDict对象
        output_dir: 可视化结果输出目录
        show_link: 是否显示本地网页链接
    
    返回:
        figures: 生成的图表对象字典
        html_links: HTML文件的本地文件系统链接列表
    """

    figures = visualize_rollout(data, num_agents, output_dir, batch_idx=batch_idx, html_file_name=html_file_name)
    
    # 只获取生成的HTML链接
    html_links = []
    html_path = os.path.join(output_dir, html_file_name)
    if os.path.exists(html_path):
        file_path = os.path.abspath(html_path)
        link = f'file://{file_path.replace(" ", "%20")}'
        html_links.append((html_file_name, link))
    
    
    # 在支持的环境中显示图表
    try:
        import plotly.io as pio
        # 设置默认渲染器
        if 'jupyterlab' in pio.renderers:
            pio.renderers.default = 'jupyterlab'
        elif 'browser' in pio.renderers:
            pio.renderers.default = 'browser'  # 使用浏览器打开
            
            # 自动打开生成的HTML文件
            import webbrowser
            webbrowser.open(html_path)
            print(f"正在浏览器中打开可视化结果...")
    except Exception as e:
        print(f"无法在当前环境中自动显示交互式图表: {e}")
        print("请打开生成的HTML文件查看")
    
    return figures, html_links

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# --- 论文绘图通用设置 ---
def set_pub_style():
    # 尝试设置字体，如果没有 Times New Roman 则回退到默认
    try:
        plt.rcParams['font.family'] = 'Times New Roman'
    except:
        pass
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    # 使用这种颜色循环，区分度高
    plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd'])
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import os
import pickle
def calc_arc_length(pos_data):
    """
    计算累计弧长
    pos_data: [time, agent, 2]
    return: [time, agent]
    """
    # 计算相邻时刻的欧氏距离
    delta = pos_data[1:] - pos_data[:-1]
    dist = np.linalg.norm(delta, axis=-1)
    # 累加得到弧长
    arc_len = np.cumsum(dist, axis=0)
    # 补上 t=0 时刻的 0
    arc_len = np.vstack([np.zeros((1, arc_len.shape[1])), arc_len])
    return arc_len
# ===================== 中文论文绘图样式配置 =====================
def set_pub_style():
    # 指定宋体字体文件路径
    font_path = '/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf'
    # 定义中文字体属性
    font_prop_chinese = fm.FontProperties(fname=font_path, size=10) 
    
    # 字体大小配置（适配单张小图）
    font_sizes = {
        'label': 10,     # 坐标轴标签
        'tick': 8,       # 刻度
        'legend': 8,     # 图例
        'title': 10      # 底部标题
    }
    
    # 全局绘图参数
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.titlepad"] = 4
    plt.rcParams["axes.labelpad"] = 4
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 0.5
    
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
    except:
        pass

    return font_prop_chinese, font_sizes

def calc_arc_length(pos_data):
    """计算累计弧长"""
    delta = pos_data[1:] - pos_data[:-1]
    dist = np.linalg.norm(delta, axis=-1)
    arc_len = np.cumsum(dist, axis=0)
    arc_len = np.vstack([np.zeros((1, arc_len.shape[1])), arc_len])
    return arc_len

def add_bottom_title_figure(fig, title_text, font_prop, font_size):
    """
    使用 Figure 坐标系在底部添加标题，避免受 Axes 缩放影响
    (0.5, 0.02) 表示画布宽度的中间，画布高度的 2% 处
    """
    fig.text(0.5, 0.02, title_text, ha='center', va='bottom', 
             fontproperties=font_prop, fontsize=font_size)

def save_single_plot(fig, output_base, suffix):
    """
    辅助函数：构造文件名并保存
    关键修复：使用 constrained_layout 自动处理布局，但要为底部标题留出空间
    """
    base, ext = os.path.splitext(output_base)
    if not ext: ext = ".pdf"
    save_path = f"{base}_{suffix}{ext}"
    
    # 保存时使用 bbox_inches='tight' 可以裁剪掉多余白边，
    # 但有时会把我们放在边缘的标题裁掉。
    # 这里我们信任 constrained_layout 的布局，不使用 bbox_inches='tight'
    # 或者，我们显式指定边距。
    
    fig.savefig(save_path, dpi=300)
    print(f"  -> 已保存: {save_path}")
    plt.close(fig)

def create_fig_and_ax(figsize):
    f, a = plt.subplots(figsize=figsize)
    # 左、右、顶 留白适应标签，底部留白给标题
    f.subplots_adjust(left=0.16, right=0.96, top=0.96, bottom=0.15)
    return f, a
def plot_straight_line_analysis(data, batch_idx=0, dt=0.05, output_path="fig_straight.pdf"):
    font_prop, fs = set_pub_style()
    print(f"开始生成直线工况分析图组 (基于 {output_path})...")
    
    valid_len = data["valid_time_steps"][batch_idx]
    time = data["time_step"][:valid_len] * dt
    pos = data["pos"][batch_idx, :valid_len, :, :]
    vel = data["vel_magnitude"][batch_idx, :valid_len, :]
    ref_vel = data["ref_vel"][batch_idx, :valid_len, 0]
    err_space = data["error_space"][batch_idx, :valid_len, :, 0]
    raw_acc = data["act_acc"][batch_idx, :valid_len, :]
    #acc = smooth_data(raw_acc, window_length=51) # 平滑处理
    acc = raw_acc # 平滑处理
    arc_len = calc_arc_length(pos)
    num_agents = pos.shape[1]

    # 使用 constrained_layout=True 自动调整布局防止重叠
    # figsize 稍微调高一点点，给底部标题留空间
    figsize = (3.5, 3.2) 

    lw_map = 1.0
    lw_curve = 1.0
    # --- 图一：累计弧长 ---
    fig, ax = create_fig_and_ax(figsize)
    for i in range(num_agents):
        ax.plot(time, arc_len[:, i]+i*10, label=f'车辆 {i}', linewidth=lw_curve)
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel('累计行程 (m)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.legend(prop=font_prop, fontsize=fs['legend'])
    #add_bottom_title_figure(fig, '(a) 累计行驶距离', font_prop, fs['title'])
    save_single_plot(fig, output_path, "distance")

    # --- 图二：速度 ---
    fig, ax = create_fig_and_ax(figsize)
    ax.plot(time, ref_vel, 'k--', label='参考速度', linewidth=lw_curve, alpha=0.8)
    for i in range(num_agents):
        ax.plot(time, vel[:, i], label=f'车辆 {i}')
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel('速度 (m/s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.legend(prop=font_prop, fontsize=fs['legend'])
    #add_bottom_title_figure(fig, '(b) 速度跟踪', font_prop, fs['title'])
    save_single_plot(fig, output_path, "velocity")

    # --- 图三：间距误差 ---
    fig, ax = create_fig_and_ax(figsize)
    for i in range(1, num_agents):
        ax.plot(time, err_space[:, i], label=f'车辆 {i}', linewidth=lw_curve)
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel('间距误差 (m)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(prop=font_prop, fontsize=fs['legend']) # 视情况开启
    #add_bottom_title_figure(fig, '(c) 间距误差', font_prop, fs['title'])
    save_single_plot(fig, output_path, "spacing_error")

    # --- 图四：加速度 ---
    fig, ax = create_fig_and_ax(figsize)
    for i in range(num_agents):
        ax.plot(time, acc[:, i], label=f'车辆 {i}', linewidth=lw_curve)
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel(r'加速度 (m/s$^2$)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.legend(prop=font_prop, fontsize=fs['legend']) # 视情况开启
    #add_bottom_title_figure(fig, '(d) 加速度曲线', font_prop, fs['title'])
    save_single_plot(fig, output_path, "acceleration")

def plot_curved_line_analysis(data, batch_idx=0, dt=0.1, output_path="fig_curve.pdf"):
    """
    第二小节：弯道工况分析 (2x2)
    map_data: 字典，需包含 'center_vertices', 'left_vertices', 'right_vertices' (numpy arrays)
    """
    font_prop, fs = set_pub_style()
    
    # 准备数据
    valid_len = data["valid_time_steps"][batch_idx]
    time = data["time_step"][:valid_len] * dt
    pos = data["pos"][batch_idx, :valid_len, :, :] 
    vel = data["vel_magnitude"][batch_idx, :valid_len, :]
    ref_vel = data["ref_vel"][batch_idx, :valid_len, 0]
    err_space = data["error_space"][batch_idx, :valid_len, :, 0]
    raw_steer = data["act_steer"][batch_idx, :valid_len, :] * (180 / np.pi)
    #steer = smooth_data(raw_steer, window_length=51) # 平滑处理
    steer = raw_steer
    num_agents = pos.shape[1]

    # 绘图
    figsize = (3.5, 3.2) 
    fig, ax = create_fig_and_ax(figsize)
    # --- 图一：弯道轨迹与跟踪效果 (XY Plot) ---
    dump_file = os.path.join("/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/debug", "map_data.pkl")
    if os.path.exists(dump_file):
        scenario_library, path_library,_,_ = pickle.load(open(dump_file, "rb"))
    else:
        raise ValueError("map_data.pkl not found in cr_map_dir")
    map_data = path_library[batch_idx]
    left = map_data.get('left_vertices').cpu().numpy()
    right = map_data.get('right_vertices').cpu().numpy()
    center = map_data.get('center_vertices').cpu().numpy()
    
    lw_map = 1.0
    lw_curve = 1.0
    if left is not None:
        ax.plot(left[:, 0], left[:, 1], color='red', linestyle='--', label='左边界', linewidth=lw_map, zorder=1)
    if right is not None:
        ax.plot(right[:, 0], right[:, 1], color='blue', linestyle='--', label='右边界', linewidth=lw_map, zorder=1)
    if center is not None:
        ax.plot(center[:, 0], center[:, 1], color='gray', linestyle='--', 
                label='中心线', linewidth=lw_map, alpha=0.5, zorder=1)

    # 绘制车辆轨迹
    for i in range(num_agents):
        ax.plot(pos[:, i, 0], pos[:, i, 1], label=f'车辆 {i}', zorder=2, linewidth=lw_curve)
        
    ax.set_xlabel('X (m)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel('Y (m)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.axis('equal')
    ax.legend(prop=font_prop, fontsize=fs['legend']) # 视情况开启
    save_single_plot(fig, output_path, "XY")

    # --- 图二：各车速度曲线 ---
    fig, ax = create_fig_and_ax(figsize)
    ax.plot(time, ref_vel, 'k--', label='参考速度', linewidth=lw_curve)
    for i in range(num_agents):
        ax.plot(time, vel[:, i], label=f'车辆 {i}')
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel('速度 (m/s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.legend(prop=font_prop, fontsize=fs['legend']) # 视情况开启
    save_single_plot(fig, output_path, "velocity")

    # --- 图三：间距误差图 (N-1条) ---
    fig, ax = create_fig_and_ax(figsize)
    for i in range(1, num_agents):
        ax.plot(time, err_space[:, i], label=f'车辆 {i}', linewidth=lw_curve)
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel('间距误差 (m)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(prop=font_prop, fontsize=fs['legend']) # 视情况开启
    save_single_plot(fig, output_path, "spacing_error")

    # --- 图四：前轮转角曲线 ---
    fig, ax = create_fig_and_ax(figsize)
    for i in range(num_agents):
        ax.plot(time, steer[:, i], label=f'车辆 {i}', linewidth=lw_curve)
    ax.set_xlabel('时间 (s)', fontproperties=font_prop, fontsize=fs['label'])
    ax.set_ylabel(r'转向角 ($^\circ$)', fontproperties=font_prop, fontsize=fs['label'])
    ax.tick_params(labelsize=fs['tick'], direction='in')
    ax.legend(prop=font_prop, fontsize=fs['legend']) # 视情况开启
    save_single_plot(fig, output_path, "steering_angle")
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd

def calculate_and_print_metrics(data, batch_idx=0, dt=0.05,output_dir_abs=None,note=""):
    """
    计算并打印多车编队的各项评测指标。
    格式统一为: 均值 (标准差, 最小值, 最大值)
    最后一行为车队整体数据的统一统计 (Unified Calculation)。
    """
    valid_len = data["valid_time_steps"][batch_idx]
    num_agents = data["vel_magnitude"].shape[2]
    
    # 提取数据 [Time, Agent]
    vel = data["vel_magnitude"][batch_idx, :valid_len, :]
    err_vel = data["error_vel"][batch_idx, :valid_len, :]
    err_space = data["error_space"][batch_idx, :valid_len, :, 0]
    acc = data["act_acc"][batch_idx, :valid_len, :]
    steer_rad = data["act_steer"][batch_idx, :valid_len, :]
    
    metrics_list = []
    
    # 存储所有车辆的原始数据，用于最后计算"车队平均"
    team_data = {
        "speed": [],
        "vel_err": [],
        "space_err": [],
        "acc": [],
        "jerk": [],
        "steer_rate": []
    }

    def get_stats_str(arr):
        """辅助函数：生成 均值 (标准差, 最小值, 最大值) 格式字符串"""
        if len(arr) == 0:
            return "0.00 (0.00, 0.00, 0.00)"
        mean_v = np.mean(arr)
        std_v = np.std(arr)
        min_v = np.min(arr)
        max_v = np.max(arr)
        return f"{mean_v:.2f} ({std_v:.2f}, {min_v:.2f}, {max_v:.2f})"

    for i in range(num_agents):
        # --- 1. 速度 (Speed) ---
        d_speed = vel[:, i]
        team_data["speed"].extend(d_speed)

        # --- 2. 速度误差 RMSE (Velocity Error) ---
        # 计算绝对误差的统计量
        d_vel_err = np.abs(err_vel[:, i])
        team_data["vel_err"].extend(d_vel_err)

        # --- 3. 间距误差 RMSE (Spacing Error) ---
        # 计算绝对误差的统计量
        d_space_err = np.abs(err_space[:, i])
        team_data["space_err"].extend(d_space_err) if i > 0 else None

        # --- 4. 加速度 (Acceleration) ---
        # 计算绝对值的统计量
        d_acc = np.abs(acc[:, i])
        team_data["acc"].extend(d_acc)

        # --- 5. Jerk (加加速度) ---
        # 差分计算 Jerk，取绝对值
        d_jerk_raw = np.diff(acc[:, i], axis=0) / dt
        d_jerk = np.abs(d_jerk_raw)
        team_data["jerk"].extend(d_jerk)

        # --- 6. 转向速率 (Steering Rate) ---
        d_steer_rate_raw = np.diff(steer_rad[:, i], axis=0) / dt
        d_steer_rate = np.abs(d_steer_rate_raw)
        team_data["steer_rate"].extend(d_steer_rate)

        # 将单车数据加入列表
        metrics_list.append({
            "车辆ID": f"车辆 {i}",
            "速度\n(m/s)": get_stats_str(d_speed),
            "速度误差RMSE\n(m/s)": get_stats_str(d_vel_err),
            "间距误差RMSE\n(m)": get_stats_str(d_space_err),
            "加速度\n(m/s^2)": get_stats_str(d_acc),
            "Jerk\n(m/s^3)": get_stats_str(d_jerk),
            "转向速率\n(rad/s)": get_stats_str(d_steer_rate)
        })
    
    # --- 计算车队平均 (Unified Calculation) ---
    # 将所有车辆的数据拼接后计算统计量
    team_row = {
        "车辆ID": "车队平均",
        "速度\n(m/s)": get_stats_str(np.array(team_data["speed"])),
        "速度误差RMSE\n(m/s)": get_stats_str(np.array(team_data["vel_err"])),
        "间距误差RMSE\n(m)": get_stats_str(np.array(team_data["space_err"])),
        "加速度\n(m/s^2)": get_stats_str(np.array(team_data["acc"])),
        "Jerk\n(m/s^3)": get_stats_str(np.array(team_data["jerk"])),
        "转向速率\n(rad/s)": get_stats_str(np.array(team_data["steer_rate"]))
    }
    metrics_list.append(team_row)

    # 创建并打印 DataFrame
    df = pd.DataFrame(metrics_list)
    if output_dir_abs is not None:
        df.to_csv(os.path.join(output_dir_abs, f"{note}_metrics_batch_{batch_idx}.csv"), index=False, encoding='utf-8-sig')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.east_asian_width', True) 
    
    print("\n" + "="*120)
    print(f" 多车编队控制性能评测指标 (Batch {batch_idx}) - 格式: 均值 (标准差, 最小值, 最大值)")
    print("="*120)
    print(df)
    print("="*120 + "\n")
    
    return df

if __name__ == "__main_ours__":
    rollout_file_path = "/home/yons/Graduation/rl_occt/outputs/2026-02-01/13-00-11_sqrt_path_reward/run-20260201_130015-9bm6xex9qe4ijxfktdio7/rollouts/rollout_iter_490_frames_29460000.pt"
    print(f"正在加载rollout文件: {rollout_file_path}")
    rollouts = load_rollout(rollout_file_path)
    output_dir = "/".join(rollout_file_path.split('/')[:-1])
    output_dir_abs = os.path.abspath(output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)

    batch_idx = 9
    data, batch_size, time_steps, num_agents = extract_rollout_data(rollouts)
    plot_straight_line_analysis(data, batch_idx=batch_idx,output_path=os.path.join(output_dir_abs, f"fig_straight_{batch_idx}.pdf"))
    metrics_df = calculate_and_print_metrics(data, batch_idx=batch_idx,output_dir_abs=output_dir_abs)

    batch_idx = 8
    data, batch_size, time_steps, num_agents = extract_rollout_data(rollouts)
    plot_curved_line_analysis(data, batch_idx=1,output_path=os.path.join(output_dir_abs, f"fig_curve_{batch_idx}.pdf")) 
    metrics_df = calculate_and_print_metrics(data, batch_idx=batch_idx,output_dir_abs=output_dir_abs)

if __name__ == "__main_chapter3_vis__":
    # ippo
    note = "ippo"
    rollout_file_path = "/home/yons/Graduation/rl_occt/outputs/platoon_comparision/16-53-02_eval_ippo_w_cp/run-20260206_165308-8d6kbm5fufhex57cn34gk/rollouts/rollout_iter_499_frames_30000000.pt"
    # mappo
    # note = "mappo"
    # rollout_file_path = "/home/yons/Graduation/rl_occt/outputs/platoon_comparision/13-56-24_eval_good_3_depth_mlp/run-20260202_135630-0elg28sk4p8icgid9wmeb/rollouts/rollout_iter_499_frames_30000000.pt"
    # mappo wo control penalty
    # note = "mappo_wo_cp"
    # rollout_file_path = "/home/yons/Graduation/rl_occt/outputs/platoon_comparision/17-01-35_eval_mappo_wo_cp/run-20260206_170139-j3ru1kkmmdg0skfn7s5ak/rollouts/rollout_iter_499_frames_30000000.pt"
    print(f"正在加载rollout文件: {rollout_file_path}")
    rollouts = load_rollout(rollout_file_path)
    output_dir = "/".join(rollout_file_path.split('/')[:-1])
    output_dir_abs = os.path.abspath(output_dir)
    os.makedirs(output_dir_abs, exist_ok=True)
    
    batch_idx = 9
    data, batch_size, time_steps, num_agents = extract_rollout_data(rollouts)
    plot_straight_line_analysis(data, batch_idx=batch_idx,output_path=os.path.join(output_dir_abs, f"fig_straight_{batch_idx}.pdf"))
    metrics_df = calculate_and_print_metrics(data, batch_idx=batch_idx,output_dir_abs=output_dir_abs,note=note)
    
    batch_idx = 8
    data, batch_size, time_steps, num_agents = extract_rollout_data(rollouts)
    plot_curved_line_analysis(data, batch_idx=1,output_path=os.path.join(output_dir_abs, f"fig_curve_{batch_idx}.pdf")) 
    metrics_df = calculate_and_print_metrics(data, batch_idx=batch_idx,output_dir_abs=output_dir_abs,note=note)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    rollout_file_path = "/home/yons/Graduation/rl_occt/outputs/2026-03-16/19-38-22/run-20260316_193825-qham7w6rbbz1huuyv01tt/rollouts/rollout_iter_499_frames_30000000.pt"
    batch_idx = 1
    try:
        print(f"正在加载rollout文件: {rollout_file_path}")
        rollouts = load_rollout(rollout_file_path)
        html_file_name=rollout_file_path.split('/')[-1].split('.')[0]+f'_batch_{batch_idx}.html'
        output_dir = "/".join(rollout_file_path.split('/')[:-1])
        output_dir_abs = os.path.abspath(output_dir)
        # 确保输出目录存在（避免可视化时创建失败）
        os.makedirs(output_dir_abs, exist_ok=True)
        data, batch_size, time_steps, num_agents = extract_rollout_data(rollouts)
        figures, html_links = visualize_your_rollout(data, num_agents, 
                                                     output_dir=output_dir_abs, 
                                                     batch_idx=batch_idx, 
                                                     html_file_name=html_file_name)
        
        print("\n可视化完成！您可以通过以下链接查看结果（Ctrl+左键点击跳转浏览器）：")
        if html_links:
            # 关键修正：提取纯文件名（去掉路径前缀）
            summary_link_entry = html_links[0][1]  # 假设原格式是 (名称, 路径/文件名)
            # 只保留文件名（不管原路径是相对还是绝对）
            summary_html_name = os.path.basename(summary_link_entry)
            
            # 启动本地HTTP服务器（--directory 指定根目录为输出目录）
            port = 8000
            # 终止占用端口的进程（Windows用taskkill，Linux/Mac用fuser）
            if os.name == "nt":  # Windows系统
                subprocess.run(
                    ["taskkill", "/f", "/im", f"python.exe", "/fi", f"pid eq {port}"],
                    capture_output=True,
                    shell=True
                )
            else:  # Linux/Mac系统
                subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
            
            # 启动服务器（--directory 指定根目录为输出目录）
            server_process = subprocess.Popen(
                [f"python3" if os.name != "nt" else "python", "-m", "http.server", str(port), "--directory", output_dir_abs],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(1)  # 延长等待时间，确保服务器完全启动
            
            # 构建正确的HTTP链接（仅包含文件名，无额外路径）
            http_link = f"http://localhost:{port}/{summary_html_name}"
            # 验证文件是否存在（避免文件名错误）
            summary_html_abs = os.path.join(output_dir_abs, summary_html_name)
            if os.path.exists(summary_html_abs):
                print(f"✅ 汇总可视化: {http_link}")
            else:
                print(f"❌ 汇总文件不存在: {summary_html_abs}")
                print(f"🔗 尝试访问目录: http://localhost:{port}/")
            
            # 输出目录的HTTP链接（直接打开所有文件）
            print(f"📂 所有文件目录: http://localhost:{port}/")
        
        # 同时保留file://链接（备用）
        output_dir_file_link = f"file://{output_dir_abs.replace(os.sep, '/')}"
        print("\n提示：关闭终端后，本地服务器会自动终止。")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
