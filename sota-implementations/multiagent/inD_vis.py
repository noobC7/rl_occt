import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# -------------------------- 核心配置（重点：简化/强化道路渲染） --------------------------
OSM_FILE_PATH = "/home/yons/commonroad/inD/maps/lanelets/01_bendplatz/location1.osm"  # 替换为你的OSM文件路径
OUTPUT_SVG_PATH = "osm_render.svg"
FIG_SIZE = (12, 10)

# 道路样式（加粗、高对比度，确保能看清）
ROAD_COLOR = "#0066CC"    # 深蓝色道路线
ROAD_LINE_WIDTH = 2       # 加粗线段（原0.8太细）
ROAD_ALPHA = 1.0          # 不透明

# 节点样式（默认关闭，先看道路）
PLOT_NODE = False
NODE_COLOR = "red"
NODE_SIZE = 1

# -------------------------- 解析OSM（放宽筛选+调试日志） --------------------------
def parse_osm(osm_file):
    """简化解析逻辑：保留所有带节点的way（不过滤highway标签），输出调试日志"""
    tree = ET.parse(osm_file)
    root = tree.getroot()

    # 1. 提取所有节点
    nodes = {}
    for node in root.findall("node"):
        node_id = node.get("id")
        try:
            lon = float(node.get("lon"))
            lat = float(node.get("lat"))
            nodes[node_id] = (lon, lat)
        except (ValueError, TypeError):
            continue  # 跳过坐标异常的节点
    print(f"解析到节点总数：{len(nodes)}")

    # 2. 提取所有way（不过滤highway，只要有有效节点就保留）
    ways = []
    empty_ways = 0  # 无有效节点的道路数
    no_tag_ways = 0 # 无highway标签的道路数
    for way in root.findall("way"):
        # 提取way的节点
        way_nodes = []
        for nd in way.findall("nd"):
            ref = nd.get("ref")
            if ref in nodes:
                way_nodes.append(nodes[ref])
        
        # 统计调试信息
        has_highway_tag = any(tag.get("k") == "highway" for tag in way.findall("tag"))
        if not has_highway_tag:
            no_tag_ways += 1
        if not way_nodes:
            empty_ways += 1
            continue  # 跳过无有效节点的道路
        
        ways.append(way_nodes)
    
    # 输出调试日志，定位问题
    print(f"解析到way总数：{len(root.findall('way'))}")
    print(f"无highway标签的way数：{no_tag_ways}")
    print(f"无有效节点的way数：{empty_ways}")
    print(f"最终可渲染的道路数：{len(ways)}")

    return nodes, ways

# -------------------------- 绘制（优先渲染道路线段） --------------------------
def plot_osm(nodes, ways):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # 1. 优先绘制道路（核心：加粗、高对比度）
    if ways:
        print(f"开始绘制 {len(ways)} 条道路...")
        for way in ways:
            way_coords = np.array(way)
            lon = way_coords[:, 0]
            lat = way_coords[:, 1]
            # 加粗线段，确保清晰可见
            ax.plot(lon, lat, color=ROAD_COLOR, linewidth=ROAD_LINE_WIDTH, alpha=ROAD_ALPHA)
    else:
        print("警告：无任何可渲染的道路！")

    # 2. 绘制节点（默认关闭，如需查看可改为True）
    if PLOT_NODE and nodes:
        node_coords = np.array(list(nodes.values()))
        ax.scatter(node_coords[:, 0], node_coords[:, 1], color=NODE_COLOR, s=NODE_SIZE, alpha=0.6)

    # 图表样式（确保道路不被遮挡）
    ax.set_aspect("equal")  # 关键：保持经纬度比例，道路不扭曲
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("OSM Roads Visualization (Only Lines)")
    ax.grid(True, linestyle="--", alpha=0.3)  # 网格透明度降低，不遮挡道路

    # 保存SVG（确保道路清晰）
    plt.tight_layout()
    plt.savefig(OUTPUT_SVG_PATH, format="svg", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSVG文件已保存至: {OUTPUT_SVG_PATH}")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    try:
        print(f"正在解析OSM文件: {OSM_FILE_PATH}")
        nodes, ways = parse_osm(OSM_FILE_PATH)
        plot_osm(nodes, ways)
    except FileNotFoundError:
        print(f"错误：未找到OSM文件 {OSM_FILE_PATH}，请检查路径！")
    except ET.ParseError:
        print("错误：OSM文件格式错误（非合法XML）！")
    except Exception as e:
        print(f"执行出错: {str(e)}")