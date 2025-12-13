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
        
        # 确保数据在CPU上
        data = {
            "time_step": np.arange(time_steps),
            "agent_id": np.arange(num_agents),
            "batch_id": np.arange(batch_size)
        }
        
        # 提取动作数据
        data["actions"] = rollouts["agents"]["action"].cpu().numpy()  # [batch, time, agent, 2]
        data["action_log_probs"] = rollouts["agents"]["action_log_prob"].cpu().numpy()  # [batch, time, agent]
        
        # 提取info中的数据
        info = rollouts["agents"]["info"]
        data["act_steer"] = info["act_steer"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["act_acc"] = info["act_acc"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["pos"] = info["pos"].cpu().numpy()  # [batch, time, agent, 2]
        data["vel"] = info["vel"].cpu().numpy()  # [batch, time, agent, 2]
        data["rot"] = info["rot"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["distance_ref"] = info["distance_ref"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["is_collision_with_agents"] = info["is_collision_with_agents"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        data["is_collision_with_lanelets"] = info["is_collision_with_lanelets"].squeeze(-1).cpu().numpy()  # [batch, time, agent]
        
        # 尝试从状态量中获取速度v，而不是计算速度大小
        # 首先检查是否存在状态量
        if "observation" in rollouts["agents"] or "state" in rollouts["agents"]:
            state_key = "observation" if "observation" in rollouts["agents"] else "state"
            state_data = rollouts["agents"][state_key].cpu().numpy()
            
            # 检查状态量维度是否为5
            if state_data.ndim == 4 and state_data.shape[-1] == 5:
                # 假设第4个维度（索引为3）是速度v
                data["vel_magnitude"] = state_data[..., 3]
                print("已从状态量中获取速度v作为速度大小")
            else:
                # 如果找不到合适的状态量，则使用原来的方法计算速度大小
                data["vel_magnitude"] = np.sqrt(data["vel"][..., 0]**2 + data["vel"][..., 1]** 2)
                print("未找到合适的状态量，使用vel的x和y分量计算速度大小")
        else:
            # 如果找不到状态量，则使用原来的方法计算速度大小
            data["vel_magnitude"] = np.sqrt(data["vel"][..., 0]**2 + data["vel"][..., 1]** 2)
            print("未找到状态量，使用vel的x和y分量计算速度大小")
        
        # 计算加速度（通过速度差分）
        acc = np.diff(data["vel"], axis=1)
        # 填充第一个时间步的加速度为0
        acc = np.concatenate([np.zeros_like(acc[:, :1]), acc], axis=1)
        data["acc"] = acc
        data["acc_magnitude"] = np.sqrt(acc[..., 0]**2 + acc[..., 1]** 2)
        
        return data, batch_size, time_steps, num_agents
    
    def plot_agent_data(self, data, batch_idx=0, agent_idx=0):
        """绘制单个agent的多种数据曲线 - 修改为2x4布局"""
        # 创建子图 - 修改为2行4列布局
        fig = make_subplots(rows=2, cols=4, 
                           subplot_titles=(
                               'Velocity Magnitude', 'Velocity Components',
                               'Acceleration Magnitude', 'Steering Angle',
                               'Distance to Reference', 'Action Components',
                               'Collisions', 'Action Log Probability'
                           ))
        
        time_steps = data["time_step"]
        
        # 1. 速度大小
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["vel_magnitude"][batch_idx, :, agent_idx],
                      mode='lines', name='Velocity Magnitude', line=dict(color='blue'),
                      legendgroup="vel_magnitude", showlegend=True),
            row=1, col=1
        )
        
        # 2. 速度分量
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["vel"][batch_idx, :, agent_idx, 0],
                      mode='lines', name='Velocity X', line=dict(color='red'),
                      legendgroup="vel_components", showlegend=True),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["vel"][batch_idx, :, agent_idx, 1],
                      mode='lines', name='Velocity Y', line=dict(color='green'),
                      legendgroup="vel_components", showlegend=True),
            row=1, col=2
        )
        
        # 3. 加速度大小
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["acc_magnitude"][batch_idx, :, agent_idx],
                      mode='lines', name='Acceleration Magnitude', line=dict(color='purple'),
                      legendgroup="acc_magnitude", showlegend=True),
            row=1, col=3
        )
        
        # 4. 转向角
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["act_steer"][batch_idx, :, agent_idx],
                      mode='lines', name='Steering Angle', line=dict(color='orange'),
                      legendgroup="steering_angle", showlegend=True),
            row=1, col=4
        )
        
        # 5. 到参考路径的距离
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["distance_ref"][batch_idx, :, agent_idx],
                      mode='lines', name='Distance to Reference', line=dict(color='brown'),
                      legendgroup="distance_ref", showlegend=True),
            row=2, col=1
        )
        
        # 6. 动作分量
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["actions"][batch_idx, :, agent_idx, 0],
                      mode='lines', name='Action 0', line=dict(color='cyan'),
                      legendgroup="action_components", showlegend=True),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["actions"][batch_idx, :, agent_idx, 1],
                      mode='lines', name='Action 1', line=dict(color='magenta'),
                      legendgroup="action_components", showlegend=True),
            row=2, col=2
        )
        
        # 7. 碰撞事件
        collision_data = data["is_collision_with_agents"][batch_idx, :, agent_idx] + \
                         data["is_collision_with_lanelets"][batch_idx, :, agent_idx]
        fig.add_trace(
            go.Scatter(x=time_steps, y=collision_data,
                      mode='lines', name='Collisions', line=dict(color='red'),
                      legendgroup="collisions", showlegend=True),
            row=2, col=3
        )
        
        # 8. 动作对数概率
        fig.add_trace(
            go.Scatter(x=time_steps, y=data["action_log_probs"][batch_idx, :, agent_idx],
                      mode='lines', name='Action Log Prob', line=dict(color='blue'),
                      legendgroup="action_log_prob", showlegend=True),
            row=2, col=4
        )
        
        fig.update_layout(
            title=f'Agent {agent_idx} Data Analysis (Batch {batch_idx})',
            height=800,  # 调整高度以适应2x4布局
            width=1600,  # 增加宽度以适应2x4布局
            hovermode='x unified'
        )
        
        # 设置每个子图的网格线
        for i in range(1, 3):
            for j in range(1, 5):
                fig.update_xaxes(title_text='Time Step', row=i, col=j, showgrid=True)
        
        return fig
    
    def create_summary_dashboard(self, data, batch_idx=0):
        """创建汇总仪表板，包含所有主要图表"""
        # 创建子图
        fig = make_subplots(rows=3, cols=2, 
                           subplot_titles=(
                               'Agent Trajectories',
                               'Velocity Comparison',
                               'Steering Angle Comparison',
                               'Distance to Reference',
                               'Collisions Over Time',
                               'Action Log Probability'
                           ))
        
        # 1. 轨迹图（简化版）
        for agent_idx in range(data["pos"].shape[2]):
            positions = data["pos"][batch_idx, :, agent_idx]
            fig.add_trace(go.Scatter(
                x=positions[:, 0], 
                y=positions[:, 1],
                mode='lines',
                name=f'Agent {agent_idx}',
                line=dict(width=2),
                legendgroup="trajectories",
                showlegend=True
            ), row=1, col=1)
        
        # 2. 速度比较
        for agent_idx in range(data["vel_magnitude"].shape[2]):
            fig.add_trace(go.Scatter(
                x=data["time_step"],
                y=data["vel_magnitude"][batch_idx, :, agent_idx],
                mode='lines',
                name=f'Agent {agent_idx}',
                line=dict(width=1.5),
                legendgroup="velocity",
                showlegend=True
            ), row=1, col=2)
        
        # 3. 转向角比较
        for agent_idx in range(data["act_steer"].shape[2]):
            fig.add_trace(go.Scatter(
                x=data["time_step"],
                y=data["act_steer"][batch_idx, :, agent_idx],
                mode='lines',
                name=f'Agent {agent_idx}',
                line=dict(width=1.5),
                legendgroup="steering",
                showlegend=True
            ), row=2, col=1)
        
        # 4. 到参考路径的距离
        for agent_idx in range(data["distance_ref"].shape[2]):
            fig.add_trace(go.Scatter(
                x=data["time_step"],
                y=data["distance_ref"][batch_idx, :, agent_idx],
                mode='lines',
                name=f'Agent {agent_idx}',
                line=dict(width=1.5),
                legendgroup="distance",
                showlegend=True
            ), row=2, col=2)
        
        # 5. 碰撞事件
        for agent_idx in range(data["is_collision_with_agents"].shape[2]):
            collision_data = data["is_collision_with_agents"][batch_idx, :, agent_idx] + \
                             data["is_collision_with_lanelets"][batch_idx, :, agent_idx]
            fig.add_trace(go.Scatter(
                x=data["time_step"],
                y=collision_data,
                mode='lines',
                name=f'Agent {agent_idx}',
                line=dict(width=1.5),
                legendgroup="collisions",
                showlegend=True
            ), row=3, col=1)
        
        # 6. 动作对数概率
        for agent_idx in range(data["action_log_probs"].shape[2]):
            fig.add_trace(go.Scatter(
                x=data["time_step"],
                y=data["action_log_probs"][batch_idx, :, agent_idx],
                mode='lines',
                name=f'Agent {agent_idx}',
                line=dict(width=1.5),
                legendgroup="log_prob",
                showlegend=True
            ), row=3, col=2)
        
        fig.update_layout(
            title=f'Rollout Summary Dashboard (Batch {batch_idx})',
            height=1500,
            width=1200,
            hovermode='x unified'
        )
        
        # 更新x轴标签
        for i in range(1, 4):
            for j in range(1, 3):
                if i > 1 or j > 1:  # 除了第一个子图
                    fig.update_xaxes(title_text='Time Step', row=i, col=j)
        
        fig.update_xaxes(title_text='X Position', row=1, col=1)
        fig.update_yaxes(title_text='Y Position', row=1, col=1)
        
        return fig
    
    def visualize_rollout(self, rollouts, output_dir="./rollout_visualizations"):
        """主可视化函数 - 简化版，只生成一个包含所有agent仪表板的HTML文件"""
        # 提取数据
        data, batch_size, time_steps, num_agents = self.extract_rollout_data(rollouts)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 对第一个批次进行可视化
        batch_idx = 0
        
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
        dashboard_div = dashboard_fig.to_html(full_html=False, include_plotlyjs=True)
        html_content += f'''
            <div class="chart-container">
                <h2>汇总仪表板</h2>
                {dashboard_div}
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
        output_path = os.path.join(output_dir, "rollout_visualization.html")
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

def visualize_your_rollout(rollouts, output_dir="./rollout_visualizations", show_link=True):
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
    figures = visualizer.visualize_rollout(rollouts, output_dir)
    
    # 只获取生成的HTML链接
    html_links = []
    html_path = os.path.join(output_dir, "rollout_visualization.html")
    if os.path.exists(html_path):
        file_path = os.path.abspath(html_path)
        link = f'file://{file_path.replace(" ", "%20")}'
        html_links.append(("rollout_visualization.html", link))
    
    
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
    rollout_file_path = "outputs/2025-12-11/22-46-06/rollouts/rollout_iter_420_frames_25260000.pt"
    
    try:
        print(f"正在加载rollout文件: {rollout_file_path}")
        rollouts = load_rollout(rollout_file_path)
        
        output_dir = f"outputs/rollout_vis/rollout_visualizations_{os.path.basename(rollout_file_path).split('.')[0]}"
        output_dir_abs = os.path.abspath(output_dir)
        # 确保输出目录存在（避免可视化时创建失败）
        os.makedirs(output_dir_abs, exist_ok=True)
        
        figures, html_links = visualize_your_rollout(rollouts, output_dir=output_dir_abs, show_link=True)
        
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