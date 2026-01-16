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
class RolloutVisualizer:
    def __init__(self):
        # 设置中文字体支持
        import plotly.io as pio
        pio.templates.default = "plotly_white"
    
    def extract_rollout_data(self, rollouts):
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
            data["reward_track_hinge"] = info["reward_track_hinge"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["vel_magnitude"] = info["vel_norm"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["ref_vel"] = info["ref_vel"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["rot"] = info["rot"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["distance_ref"] = info["distance_ref"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
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
        # penalty_collide_with_agents: -100
        # penalty_outside_boundaries: -100
        return data, batch_size, time_steps, num_agents
    
    def plot_agent_data(self, data, batch_idx=0, agent_idx=0):
        # 重新组织布局：将reward和penalty分别集中显示
        # Row 1: 状态变量 (6列)
        # Row 2: Rewards (2列，每列最多4条曲线)
        # Row 3: Penalties (2列，每列最多4条曲线)
        # Row 4: Hinge相关 (3列，如果有)

        has_hinge = "hinge_dis" in data.keys()
        num_rows = 2
        fix_titles=[
                'Speed[m/s]', 'Acceleration[m/s^2]', 'Heading[degree]',
                'Steering Angle[degree]', 'Distance to Reference[m]', 'Space Error[m]',
                'Rewards Total','Rewards Group 1', 'Rewards Group 2',
                'Penalties Group 1', 'Penalties Group 2',
                'Action Log Prob'
                ]
        hinge_titles=['Hinge Dis[m]', 'Hinge Status', 'Reward Track Hinge']
        fig = make_subplots(
            rows=num_rows, cols=7,
            subplot_titles=fix_titles+hinge_titles if has_hinge else fix_titles
        )

        valid_time_steps = data["valid_time_steps"][batch_idx]
        time_steps = data["time_step"][:valid_time_steps]

        color_list=["blue","purple","green","orange","brown","red","black","cyan","magenta","gray","olive","pink","teal","navy","salmon","turquoise"]

        # ============ Row 1: 状态变量 ============
        # Speed, Ref Vel, Vel Error
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["vel_magnitude"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Speed', line=dict(color=color_list[0]),
                    legendgroup="speed", showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["ref_vel"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Ref Vel', line=dict(color=color_list[10]),
                    legendgroup="ref_vel", showlegend=True),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["error_vel"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Vel Error', line=dict(color=color_list[5]),
                    legendgroup="vel_error", showlegend=True),
            row=1, col=1
        )

        # Acceleration
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["act_acc"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Acceleration', line=dict(color=color_list[1]),
                    legendgroup="acceleration", showlegend=True),
            row=1, col=2
        )

        # Heading
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["rot"][batch_idx, :valid_time_steps, agent_idx]/np.pi*180,
                    mode='lines', name='Heading', line=dict(color=color_list[2]),
                    legendgroup="heading_angle", showlegend=True),
            row=1, col=3
        )

        # Steering Angle
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["act_steer"][batch_idx, :valid_time_steps, agent_idx]/np.pi*180,
                    mode='lines', name='Steering Angle', line=dict(color=color_list[3]),
                    legendgroup="steering_angle", showlegend=True),
            row=1, col=4
        )

        # Distance to Reference
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["distance_ref"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Distance to Reference', line=dict(color=color_list[4]),
                    legendgroup="distance_ref", showlegend=True),
            row=1, col=5
        )

        # Space Error (Front & Back)
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["error_space"][batch_idx, :valid_time_steps, agent_idx,0],
                    mode='lines', name='Space Front', line=dict(color=color_list[6]),
                    legendgroup="space_front", showlegend=True),
            row=1, col=6
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["error_space"][batch_idx, :valid_time_steps, agent_idx,1],
                    mode='lines', name='Space Back', line=dict(color=color_list[7]),
                    legendgroup="space_back", showlegend=True),
            row=1, col=6
        )

        # ============ Row 2: Rewards ============
        # Rewards Group 1: Total, Progress, Vel, Goal (在同一图中)
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_total"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Total', line=dict(color=color_list[0]),
                    legendgroup="reward_total", showlegend=True),
            row=1, col=7
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_progress"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Progress', line=dict(color=color_list[1]),
                    legendgroup="reward_progress", showlegend=True),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_vel"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Vel', line=dict(color=color_list[2]),
                    legendgroup="reward_vel", showlegend=True),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_goal"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Goal', line=dict(color=color_list[3]),
                    legendgroup="reward_goal", showlegend=True),
            row=2, col=1
        )
        # Rewards Group 2: Track Ref Vel, Space, Heading, Path (在同一图中)
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_track_ref_vel"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Track Ref Vel', line=dict(color=color_list[4]),
                    legendgroup="reward_track_ref_vel", showlegend=True),
            row=2, col=1
        )

        if has_hinge:
            fig.add_trace(
                go.Scatter(x=time_steps, y=data["reward_track_hinge"][batch_idx, :valid_time_steps, agent_idx],
                        mode='lines', name='Reward Track Hinge', line=dict(color=color_list[2]),
                        legendgroup="reward_track_hinge", showlegend=True),
                row=2, col=2
            )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_track_ref_space"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Track Ref Space', line=dict(color=color_list[5]),
                    legendgroup="reward_track_ref_space", showlegend=True),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_track_ref_heading"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Track Ref Heading', line=dict(color=color_list[6]),
                    legendgroup="reward_track_ref_heading", showlegend=True),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["reward_track_ref_path"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Reward Track Ref Path', line=dict(color=color_list[7]),
                    legendgroup="reward_track_ref_path", showlegend=True),
            row=2, col=2
        )

        # ============ Row 3: Penalties ============
        # Penalties Group 1: Change Steering, Change Acc, Action Log Prob (在同一图中)
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["penalty_change_steering"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Penalty Change Steering', line=dict(color=color_list[8]),
                    legendgroup="penalty_change_steering", showlegend=True),
            row=2, col=3
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["penalty_change_acc"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Penalty Change Acc', line=dict(color=color_list[9]),
                    legendgroup="penalty_change_acc", showlegend=True),
            row=2, col=3
        )

        # Penalties Group 2: Collide with Agents, Outside Boundaries, Near Boundary, Near Other Agents (在同一图中)
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["penalty_collide_with_agents"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Penalty Collide with Agents', line=dict(color=color_list[11]),
                    legendgroup="penalty_collide_with_agents", showlegend=True),
            row=2, col=4
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["penalty_outside_boundaries"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Penalty Outside Boundaries', line=dict(color=color_list[12]),
                    legendgroup="penalty_outside_boundaries", showlegend=True),
            row=2, col=4
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["penalty_near_boundary"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Penalty Near Boundary', line=dict(color=color_list[13]),
                    legendgroup="penalty_near_boundary", showlegend=True),
            row=2, col=4
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["penalty_near_other_agents"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Penalty Near Other Agents', line=dict(color=color_list[14]),
                    legendgroup="penalty_near_other_agents", showlegend=True),
            row=2, col=4
        )

        fig.add_trace(
            go.Scatter(x=time_steps, y=data["action_log_probs"][batch_idx, :valid_time_steps, agent_idx],
                    mode='lines', name='Action Log Prob', line=dict(color=color_list[10]),
                    legendgroup="action_log_prob", showlegend=True),
            row=2, col=5
        )
        # ============ Row 4: Hinge相关 (如果有) ============
        if has_hinge:
            # Hinge Dis
            fig.add_trace(
                go.Scatter(x=time_steps, y=data["hinge_dis"][batch_idx, :valid_time_steps, agent_idx],
                        mode='lines', name='Hinge Dis', line=dict(color=color_list[0]),
                        legendgroup="hinge_dis", showlegend=True),
                row=2, col=6
            )
            # Hinge Status
            fig.add_trace(
                go.Scatter(x=time_steps, y=data["hinge_status"][batch_idx, :valid_time_steps, agent_idx],
                        mode='lines', name='Hinge Status', line=dict(color=color_list[1]),
                        legendgroup="hinge_status", showlegend=True),
                row=2, col=7
            )

        fig.update_layout(
            title=f'Agent {agent_idx} Data Analysis (Batch {batch_idx})',
            height=400 * num_rows,  # 根据行数调整高度
            width=2200,
            hovermode='x unified'
        )

        return fig
    def create_summary_dashboard(self, data, batch_idx=0):
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
    
    def visualize_rollout(self, rollouts, output_dir="./rollout_visualizations", batch_idx=0, html_file_name="rollout_visualization.html"):
        """主可视化函数 - 简化版，只生成一个包含所有agent仪表板的HTML文件"""
        # 提取数据
        data, batch_size, time_steps, num_agents = self.extract_rollout_data(rollouts)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建一个大的图表，包含汇总仪表板和所有agent的详细仪表板
        from plotly import subplots
        
        # 计算需要的行数：1行用于汇总仪表板，每个agent占1行
        total_rows = 1 + num_agents
        
        # 创建一个大的图表容器
        main_fig = go.Figure()
        
        # 生成HTML内容，将所有图表嵌入到一个HTML文件中
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
        dashboard_fig = self.create_summary_dashboard(data, batch_idx=batch_idx)
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
            agent_fig = self.plot_agent_data(data, batch_idx=batch_idx, agent_idx=agent_idx)
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

def visualize_your_rollout(rollouts, output_dir="./rollout_visualizations", batch_idx=0, html_file_name="rollout_visualization.html"):
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
    # 创建可视化器实例
    visualizer = RolloutVisualizer()
    
    # 执行可视化
    figures = visualizer.visualize_rollout(rollouts, output_dir, batch_idx=batch_idx, html_file_name=html_file_name)
    
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

if __name__ == "__main__":
    
    rollout_file_path = "/home/yons/Graduation/rl_occt/outputs/2026-01-16/15-58-02/run-20260116_155804-hid8z7kc1qe8i0rt6fzr2/rollouts/rollout_iter_40_frames_2460000.pt"
    batch_idx = 0
    try:
        print(f"正在加载rollout文件: {rollout_file_path}")
        rollouts = load_rollout(rollout_file_path)
        html_file_name=rollout_file_path.split('/')[-1].split('.')[0]+f'_batch_{batch_idx}.html'
        output_dir = "/".join(rollout_file_path.split('/')[:-1])
        output_dir_abs = os.path.abspath(output_dir)
        # 确保输出目录存在（避免可视化时创建失败）
        os.makedirs(output_dir_abs, exist_ok=True)
        
        figures, html_links = visualize_your_rollout(rollouts, 
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