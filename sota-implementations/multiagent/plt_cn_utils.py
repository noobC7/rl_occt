import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# ===================== 中文论文绘图样式配置（对齐参考代码）=====================
# 指定宋体字体文件路径（参考代码中的路径）
font_path = '/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf'
# 定义中文字体属性（宋体，适配小尺寸论文插图）
font_prop_chinese = fm.FontProperties(fname=font_path, size=7)  # 小尺寸适配
# 字体大小统一配置（适配小尺寸论文插图）
font_size_label = 7     # 坐标轴标签字体大小（适配3x2画布）
font_size_tick = 6      # 刻度字体大小（数字用新罗马）
font_size_legend = 6    # 图例字体大小
font_size_title = 8     # 标题字体大小

# 全局绘图参数（极致适配小尺寸论文插图+中文显示）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常问题
plt.rcParams["figure.dpi"] = 300            # 高清分辨率，确保缩小后仍清晰
plt.rcParams["legend.frameon"] = False      # 去除图例边框，节省空间
plt.rcParams["axes.titlepad"] = 4           # 减小标题与坐标轴间距
plt.rcParams["axes.labelpad"] = 2           # 减小标签与坐标轴间距
# plt.rcParams["grid.alpha"] = 0.3            # 网格透明度（参考代码样式）
# plt.rcParams["grid.linestyle"] = "--"        # 网格线型（参考代码样式）
# plt.rcParams["grid.linewidth"] = 0.5        # 网格线宽（参考代码样式）
plt.rcParams['axes.grid'] = False
