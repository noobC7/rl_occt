# -*- coding: utf-8 -*-
"""
奖励函数设计消融可视化代码
功能：通过 SwanLab OpenApi 读取奖励函数消融实验数据，绘制关键指标对比曲线
适用场景：硕士毕业设计中奖励函数必要性分析与可视化展示
"""

from plt_cn_utils import *
from swanlab import OpenApi
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


PROJECT_NAME = "MAPPO_OCCT_REWARD_ABLATION"
SAVE_DIR = "./outputs/reward_ablation_vis/"
CUT_ITER = None
EMA_ALPHA = 0.05
FLUCTUATION_ALPHA = 0.20
FLUCTUATION_SCALE = 1.20
DRAW_RAW_CURVE = False

# 适配 A4 页面正文区域单排 1x3 摆放的小尺寸论文插图
# 宽度约 4.9 cm，可在常见页边距下较自然地横向排下 3 张图
FIG_SIZE = (1.92, 1.50)
LOCAL_FONT_SIZE_LABEL = 8
LOCAL_FONT_SIZE_TICK = 8
LOCAL_FONT_SIZE_LEGEND = 8

AXIS_FONT_PROP = fm.FontProperties(fname=font_path, size=LOCAL_FONT_SIZE_LABEL)
LEGEND_FONT_PROP = fm.FontProperties(fname=font_path, size=LOCAL_FONT_SIZE_LEGEND)

EXPERIMENT_SPECS = [
    {
        "exp_id": "ohc8va4sd332ujsao4jn0",
        "label": "Baseline",
    },
    {
        "exp_id": "v36aaiq0gjjq4iras5yqe",
        "label": "w/o TimeCost",
    },
    {
        "exp_id": "cxuktsnie7rc59hvp0em7",
        "label": "w/o HoldReward",
    },
]

METRIC_SPECS = [
    {
        "metric": "info/hinge_steps",
        "y_label": "铰接步数",
        "file_name": "reward_ablation_hinge_steps.pdf",
        "clip": (0.0, None),
        "colors": ["#1D3557", "#5BB8F3", "#0B9486"],
    },
    {
        "metric": "info/done_all_hinged_rate",
        "y_label": "完全铰接率",
        "file_name": "reward_ablation_done_all_hinged_rate.pdf",
        "clip": (0.0, 1.0),
        "yticks": [0.0, 1.0],
        "colors": ["#B45415", "#9E9204", "#689C00"],
    },
    {
        "metric": "info/done_collision_with_exit_segments_rate",
        "y_label": "超时未铰接率",
        "file_name": "reward_ablation_done_collision_with_exit_segments_rate.pdf",
        "clip": (0.0, 1.0),
        "yticks": [0.0, 1.0],
        "colors": ["#6602A9", "#C71D97", "#BB080E"],
    },
]


def _slice_series(series):
    if CUT_ITER is None:
        return series
    return series.iloc[:CUT_ITER]


def _ema_smooth(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    if values.size == 0:
        return values

    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def _apply_clip(values: np.ndarray, clip_range: tuple[float | None, float | None] | None):
    if clip_range is None:
        return values

    clipped_values = values.copy()
    lower_bound, upper_bound = clip_range
    if lower_bound is not None:
        clipped_values = np.maximum(clipped_values, lower_bound)
    if upper_bound is not None:
        clipped_values = np.minimum(clipped_values, upper_bound)
    return clipped_values


def _compute_fluctuation_bounds(
    raw_values: np.ndarray,
    center_values: np.ndarray,
    alpha: float = FLUCTUATION_ALPHA,
    scale: float = FLUCTUATION_SCALE,
    clip_range: tuple[float | None, float | None] | None = None,
):
    deviation = np.abs(raw_values - center_values)
    band_width = _ema_smooth(deviation, alpha=alpha) * scale
    lower_values = center_values - band_width
    upper_values = center_values + band_width

    return (
        _apply_clip(lower_values, clip_range),
        _apply_clip(upper_values, clip_range),
    )


def _metric_candidates(metric_name: str) -> list[str]:
    normalized_metric = metric_name.replace("/", "_")
    candidates = [
        metric_name,
        normalized_metric,
        f"train_{metric_name}",
        f"train_{normalized_metric}",
        f"eval_{metric_name}",
        f"eval_{normalized_metric}",
    ]

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _extract_series_from_metric_frame(metric_frame, metric_key: str):
    if metric_frame is None or not hasattr(metric_frame, "empty") or metric_frame.empty:
        return None
    if not hasattr(metric_frame, "columns"):
        return None

    if metric_key in metric_frame.columns:
        target_series = metric_frame[metric_key]
    elif len(metric_frame.columns) == 1:
        target_series = metric_frame.iloc[:, 0]
        target_series.name = metric_key
    else:
        return None

    target_series = _slice_series(target_series).dropna()
    if target_series.empty:
        return None
    return target_series


def _fetch_metric_series(
    swanlab_api: OpenApi,
    exp_id: str,
    metric_name: str,
):
    last_error = None

    for candidate in _metric_candidates(metric_name):
        try:
            metric_frame = swanlab_api.get_metrics(exp_id=exp_id, keys=[candidate]).data
        except Exception as exc:
            last_error = exc
            continue

        target_series = _extract_series_from_metric_frame(metric_frame, candidate)
        if target_series is not None:
            return candidate, target_series

    raise ValueError(
        f"实验 {exp_id} 无法读取指标 {metric_name}。"
        f"{'' if last_error is None else f' 最近一次错误：{last_error}'}"
    )


def _style_axes(ax, y_label: str, y_ticks: list[float] | None = None):
    # ax.set_ylabel(y_label, fontproperties=AXIS_FONT_PROP, fontsize=LOCAL_FONT_SIZE_LABEL)
    ax.set_xlabel("训练轮数", fontproperties=AXIS_FONT_PROP, fontsize=LOCAL_FONT_SIZE_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    if y_ticks is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    else:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(int(tick)) for tick in y_ticks])
    ax.tick_params(
        axis="both",
        labelsize=LOCAL_FONT_SIZE_TICK,
        pad=1,
        direction="in",
        top=False,
        right=False,
        labelfontfamily="Times New Roman",
    )
    ax.margins(x=0)
    ax.legend(
        loc="best",
        prop=LEGEND_FONT_PROP,
        handlelength=1.2,
        borderpad=0.15,
        labelspacing=0.2,
    )


def _save_figure(file_name: str):
    plt.savefig(
        os.path.join(SAVE_DIR, file_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
    )
    plt.close()


def plot_metric_ablation(
    swanlab_api: OpenApi,
    metric_name: str,
    y_label: str,
    file_name: str,
    clip_range: tuple[float | None, float | None] | None = None,
    line_colors: list[str] | None = None,
    y_ticks: list[float] | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
    resolved_metric_names = []

    for experiment_idx, experiment_spec in enumerate(EXPERIMENT_SPECS):
        color = (
            line_colors[experiment_idx]
            if line_colors is not None and experiment_idx < len(line_colors)
            else "#4E79A7"
        )
        resolved_metric_name, target_series = _fetch_metric_series(
            swanlab_api=swanlab_api,
            exp_id=experiment_spec["exp_id"],
            metric_name=metric_name,
        )
        resolved_metric_names.append(
            (experiment_spec["label"], experiment_spec["exp_id"], resolved_metric_name)
        )

        x_axis = target_series.index.to_numpy()
        raw_values = target_series.to_numpy(dtype=float)
        smooth_values = _ema_smooth(raw_values)
        lower_values, upper_values = _compute_fluctuation_bounds(
            raw_values=raw_values,
            center_values=smooth_values,
            clip_range=clip_range,
        )

        if DRAW_RAW_CURVE:
            ax.plot(
                x_axis,
                raw_values,
                linewidth=0.55,
                color=color,
                alpha=0.16,
            )

        ax.plot(
            x_axis,
            upper_values,
            linewidth=0.40,
            color=color,
            alpha=0.30,
        )
        ax.plot(
            x_axis,
            lower_values,
            linewidth=0.40,
            color=color,
            alpha=0.30,
        )
        ax.fill_between(
            x_axis,
            lower_values,
            upper_values,
            color=color,
            alpha=0.12,
        )
        ax.plot(
            x_axis,
            smooth_values,
            label=experiment_spec["label"],
            linewidth=1.0,
            color=color,
            alpha=0.95,
        )

    if clip_range is not None and all(bound is not None for bound in clip_range):
        ax.set_ylim(clip_range)

    _style_axes(ax, y_label, y_ticks=y_ticks)
    _save_figure(file_name)

    print(f"{file_name} 指标解析结果：")
    for label, exp_id, resolved_metric_name in resolved_metric_names:
        print(f"  - {label} ({exp_id}) -> {resolved_metric_name}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    swanlab_api = OpenApi()
    project_experiments = swanlab_api.list_experiments(project=PROJECT_NAME).data
    print(f"成功获取项目 {PROJECT_NAME} 下 {len(project_experiments)} 个实验数据")

    for metric_spec in METRIC_SPECS:
        plot_metric_ablation(
            swanlab_api=swanlab_api,
            metric_name=metric_spec["metric"],
            y_label=metric_spec["y_label"],
            file_name=metric_spec["file_name"],
            clip_range=metric_spec.get("clip"),
            line_colors=metric_spec.get("colors"),
            y_ticks=metric_spec.get("yticks"),
        )
