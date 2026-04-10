# -*- coding: utf-8 -*-
"""
Failure Curriculum 训练机制可视化代码
功能：通过 SwanLab OpenApi 读取 failure curriculum 相关训练指标，绘制 3 张论文风格 PDF 图
适用场景：说明 replay curriculum 机制是否按预期启动、被使用以及是否在消化 hard cases
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from swanlab import OpenApi

from plt_cn_utils import *


PROJECT_NAME = "MAPPO_OCCT_ROUNDABOUT_EXTEND"
BASELINE_EXPERIMENT_ID = "4y6fhik52k0czic9vpa4j"
CURRICULUM_EXPERIMENT_ID = "c4x3i37ud5ujzpdekff6f"
SAVE_DIR = "./outputs/failure_curriculum_vis/"

CUT_ITER = None
FLUCTUATION_ALPHA = 0.20
FLUCTUATION_SCALE = 1.20
DRAW_RAW_CURVE = False

# 与 reward_ablation_vis_cn.py 保持一致
FIG_SIZE = (1.92, 1.50)
LOCAL_FONT_SIZE_LABEL = 8
LOCAL_FONT_SIZE_TICK = 8
LOCAL_FONT_SIZE_LEGEND = 8

AXIS_FONT_PROP = fm.FontProperties(fname=font_path, size=LOCAL_FONT_SIZE_LABEL)
LEGEND_FONT_PROP = fm.FontProperties(fname=font_path, size=LOCAL_FONT_SIZE_LEGEND)


METRIC_SPECS = [
    {
        "metric": "failure_curriculum/replay_probability",
        "y_label": "重放概率",
        "file_name": "failure_curriculum_replay_probability.pdf",
        "clip": (0.0, 0.32),
        "yticks": [0.0, 0.1, 0.2, 0.3],
        "color": "#C56A1A",
        "label": "重放概率",
    },
    {
        "metric": "failure_curriculum/active_replays",
        "y_label": "活跃重放数",
        "file_name": "failure_curriculum_active_replays.pdf",
        "clip": (0.0, None),
        "color": "#007639",
        "label": "活跃重放数",
    },
]

DUAL_METRIC_SPEC = {
    "left_metric": "failure_curriculum/total_removed_success",
    "right_metric": "failure_curriculum/total_updated_failure",
    "left_label": "成功移除",
    "right_label": "失败更新",
    "file_name": "failure_curriculum_removed_vs_updated.pdf",
    "left_color": "#2B6CB0",
    "right_color": "#C05621",
}

COMPARISON_METRIC_SPECS = [
    {
        "metric": "road/road_0_mean_step",
        "y_label": "Road0平均步数",
        "file_name": "failure_curriculum_compare_road_0_mean_step.pdf",
        "baseline_color": "#C125AA",
        "curriculum_color": "#EC8F0E",
    },
    {
        "metric": "default/iteration_time",
        "y_label": "单轮训练时间(s)",
        "file_name": "failure_curriculum_compare_iteration_time.pdf",
        "baseline_color": "#B73A3A",
        "curriculum_color": "#13BA1E",
    },
    {
        "metric": "info/success_rate",
        "y_label": "成功率",
        "file_name": "failure_curriculum_compare_success_rate.pdf",
        "clip": (0.0, 1.0),
        "yticks": [0.0, 0.5, 1.0],
        "baseline_color": "#6C8E2A",
        "curriculum_color": "#5C3D99",
    },
    {
        "metric": "reward/episode_reward_mean",
        "y_label": "平均回合奖励",
        "file_name": "failure_curriculum_compare_episode_reward_mean.pdf",
        "baseline_color": "#FDB03D",
        "curriculum_color": "#74C21A",
    },
]
COMPARISON_BASELINE_LABEL = "IPPO"
COMPARISON_CURRICULUM_LABEL = "IPPO w/ FC"


def _safe_get_attr(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _slice_series(series):
    if CUT_ITER is None:
        return series
    return series.iloc[:CUT_ITER]


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
    band_width = deviation * scale
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
        f"train/{metric_name}",
        f"train_{normalized_metric}",
        f"train/{normalized_metric}",
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


def _resolve_experiment(
    swanlab_api: OpenApi, exp_id: str
) -> tuple[str, str]:
    experiments = swanlab_api.list_experiments(project=PROJECT_NAME).data
    if experiments is None:
        return exp_id, exp_id

    for experiment in experiments:
        current_exp_id = _safe_get_attr(experiment, "id", None)
        if current_exp_id is None:
            continue
        if str(current_exp_id) == exp_id:
            return exp_id, str(_safe_get_attr(experiment, "name", exp_id))

    return exp_id, exp_id


def _style_axes(ax, y_label: str, y_ticks: list[float] | None = None):
    # 与 reward_ablation_vis_cn.py 保持一致，不额外加标题和 y 标签
    ax.set_xlabel("训练轮数", fontproperties=AXIS_FONT_PROP, fontsize=LOCAL_FONT_SIZE_LABEL)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    if y_ticks is None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    else:
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(tick) for tick in y_ticks])
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


def _style_axis_color(ax, color: str, side: str = "left"):
    if side == "left":
        ax.tick_params(axis="y", colors=color)
        ax.spines["left"].set_color(color)
    elif side == "right":
        ax.tick_params(axis="y", colors=color)
        ax.spines["right"].set_color(color)


def _save_figure(file_name: str):
    plt.savefig(
        os.path.join(SAVE_DIR, file_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.03,
    )
    plt.close()


def plot_single_metric(
    swanlab_api: OpenApi,
    exp_id: str,
    exp_label: str,
    *,
    metric_name: str,
    y_label: str,
    file_name: str,
    color: str,
    legend_label: str,
    clip_range: tuple[float | None, float | None] | None = None,
    y_ticks: list[float] | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
    resolved_metric_name, target_series = _fetch_metric_series(
        swanlab_api=swanlab_api,
        exp_id=exp_id,
        metric_name=metric_name,
    )

    x_axis = target_series.index.to_numpy()
    raw_values = target_series.to_numpy(dtype=float)
    smooth_values = raw_values.copy()
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
        label=legend_label,
        linewidth=1.0,
        color=color,
        alpha=0.95,
    )

    if clip_range is not None and all(bound is not None for bound in clip_range):
        ax.set_ylim(clip_range)

    _style_axes(ax, y_label, y_ticks=y_ticks)
    _save_figure(file_name)

    print(f"{file_name} 指标解析结果：")
    print(f"  - {exp_label} -> {resolved_metric_name}")


def plot_dual_metric(
    swanlab_api: OpenApi,
    exp_id: str,
    exp_label: str,
    *,
    left_metric: str,
    right_metric: str,
    left_label: str,
    right_label: str,
    file_name: str,
    left_color: str,
    right_color: str,
):
    fig, ax_left = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)
    ax_right = ax_left.twinx()

    resolved_left_metric, left_series = _fetch_metric_series(
        swanlab_api=swanlab_api,
        exp_id=exp_id,
        metric_name=left_metric,
    )
    resolved_right_metric, right_series = _fetch_metric_series(
        swanlab_api=swanlab_api,
        exp_id=exp_id,
        metric_name=right_metric,
    )

    left_x = left_series.index.to_numpy()
    left_raw = left_series.to_numpy(dtype=float)
    left_smooth = left_raw.copy()
    left_low, left_up = _compute_fluctuation_bounds(
        raw_values=left_raw,
        center_values=left_smooth,
    )

    right_x = right_series.index.to_numpy()
    right_raw = right_series.to_numpy(dtype=float)
    right_smooth = right_raw.copy()
    right_low, right_up = _compute_fluctuation_bounds(
        raw_values=right_raw,
        center_values=right_smooth,
    )

    ax_left.plot(left_x, left_up, linewidth=0.40, color=left_color, alpha=0.30)
    ax_left.plot(left_x, left_low, linewidth=0.40, color=left_color, alpha=0.30)
    ax_left.fill_between(left_x, left_low, left_up, color=left_color, alpha=0.12)
    left_line = ax_left.plot(
        left_x,
        left_smooth,
        label=left_label,
        linewidth=1.0,
        color=left_color,
        alpha=0.95,
    )[0]

    ax_right.plot(right_x, right_up, linewidth=0.40, color=right_color, alpha=0.30)
    ax_right.plot(right_x, right_low, linewidth=0.40, color=right_color, alpha=0.30)
    ax_right.fill_between(right_x, right_low, right_up, color=right_color, alpha=0.10)
    right_line = ax_right.plot(
        right_x,
        right_smooth,
        label=right_label,
        linewidth=1.0,
        color=right_color,
        alpha=0.95,
    )[0]

    _style_axes(ax_left, left_label)
    _style_axis_color(ax_left, left_color, side="left")
    ax_right.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_right.tick_params(
        axis="y",
        labelsize=LOCAL_FONT_SIZE_TICK,
        pad=1,
        direction="in",
        right=True,
        labelfontfamily="Times New Roman",
        colors=right_color,
    )
    _style_axis_color(ax_right, right_color, side="right")

    ax_left.legend(
        handles=[left_line, right_line],
        labels=[left_label, right_label],
        loc="best",
        prop=LEGEND_FONT_PROP,
        handlelength=1.2,
        borderpad=0.15,
        labelspacing=0.2,
    )
    _save_figure(file_name)

    print(f"{file_name} 指标解析结果：")
    print(f"  - {exp_label} -> {resolved_left_metric}")
    print(f"  - {exp_label} -> {resolved_right_metric}")


def plot_comparison_metric(
    swanlab_api: OpenApi,
    *,
    baseline_exp_id: str,
    baseline_exp_label: str,
    curriculum_exp_id: str,
    curriculum_exp_label: str,
    metric_name: str,
    y_label: str,
    file_name: str,
    baseline_color: str,
    curriculum_color: str,
    baseline_label: str,
    curriculum_label: str,
    clip_range: tuple[float | None, float | None] | None = None,
    y_ticks: list[float] | None = None,
):
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, constrained_layout=True)

    resolved_baseline_metric, baseline_series = _fetch_metric_series(
        swanlab_api=swanlab_api,
        exp_id=baseline_exp_id,
        metric_name=metric_name,
    )
    resolved_curriculum_metric, curriculum_series = _fetch_metric_series(
        swanlab_api=swanlab_api,
        exp_id=curriculum_exp_id,
        metric_name=metric_name,
    )

    baseline_x = baseline_series.index.to_numpy()
    baseline_raw = baseline_series.to_numpy(dtype=float)
    baseline_low, baseline_up = _compute_fluctuation_bounds(
        raw_values=baseline_raw,
        center_values=baseline_raw,
    )

    curriculum_x = curriculum_series.index.to_numpy()
    curriculum_raw = curriculum_series.to_numpy(dtype=float)
    curriculum_low, curriculum_up = _compute_fluctuation_bounds(
        raw_values=curriculum_raw,
        center_values=curriculum_raw,
    )

    ax.plot(
        baseline_x,
        baseline_up,
        linewidth=0.40,
        color=baseline_color,
        alpha=0.28,
    )
    ax.plot(
        baseline_x,
        baseline_low,
        linewidth=0.40,
        color=baseline_color,
        alpha=0.28,
    )
    ax.fill_between(
        baseline_x,
        baseline_low,
        baseline_up,
        color=baseline_color,
        alpha=0.10,
    )
    ax.plot(
        baseline_x,
        baseline_raw,
        label=baseline_label,
        linewidth=1.0,
        color=baseline_color,
        alpha=0.95,
    )

    ax.plot(
        curriculum_x,
        curriculum_up,
        linewidth=0.40,
        color=curriculum_color,
        alpha=0.28,
    )
    ax.plot(
        curriculum_x,
        curriculum_low,
        linewidth=0.40,
        color=curriculum_color,
        alpha=0.28,
    )
    ax.fill_between(
        curriculum_x,
        curriculum_low,
        curriculum_up,
        color=curriculum_color,
        alpha=0.10,
    )
    ax.plot(
        curriculum_x,
        curriculum_raw,
        label=curriculum_label,
        linewidth=1.0,
        color=curriculum_color,
        alpha=0.95,
    )

    if clip_range is not None and all(bound is not None for bound in clip_range):
        ax.set_ylim(clip_range)

    _style_axes(ax, y_label, y_ticks=y_ticks)
    ax.legend(
        loc="best",
        prop=LEGEND_FONT_PROP,
        handlelength=1.2,
        borderpad=0.15,
        labelspacing=0.2,
    )
    _save_figure(file_name)

    print(f"{file_name} 指标解析结果：")
    print(f"  - {baseline_exp_label} -> {resolved_baseline_metric}")
    print(f"  - {curriculum_exp_label} -> {resolved_curriculum_metric}")


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    swanlab_api = OpenApi()
    curriculum_exp_id, curriculum_exp_name = _resolve_experiment(
        swanlab_api, CURRICULUM_EXPERIMENT_ID
    )
    baseline_exp_id, baseline_exp_name = _resolve_experiment(
        swanlab_api, BASELINE_EXPERIMENT_ID
    )
    print(f"使用课程实验：{curriculum_exp_name} (exp_id={curriculum_exp_id})")
    print(f"使用基线实验：{baseline_exp_name} (exp_id={baseline_exp_id})")

    for metric_spec in METRIC_SPECS:
        plot_single_metric(
            swanlab_api=swanlab_api,
            exp_id=curriculum_exp_id,
            exp_label=curriculum_exp_name,
            metric_name=metric_spec["metric"],
            y_label=metric_spec["y_label"],
            file_name=metric_spec["file_name"],
            color=metric_spec["color"],
            legend_label=metric_spec["label"],
            clip_range=metric_spec.get("clip"),
            y_ticks=metric_spec.get("yticks"),
        )

    plot_dual_metric(
        swanlab_api=swanlab_api,
        exp_id=curriculum_exp_id,
        exp_label=curriculum_exp_name,
        left_metric=DUAL_METRIC_SPEC["left_metric"],
        right_metric=DUAL_METRIC_SPEC["right_metric"],
        left_label=DUAL_METRIC_SPEC["left_label"],
        right_label=DUAL_METRIC_SPEC["right_label"],
        file_name=DUAL_METRIC_SPEC["file_name"],
        left_color=DUAL_METRIC_SPEC["left_color"],
        right_color=DUAL_METRIC_SPEC["right_color"],
    )

    for comparison_spec in COMPARISON_METRIC_SPECS:
        plot_comparison_metric(
            swanlab_api=swanlab_api,
            baseline_exp_id=baseline_exp_id,
            baseline_exp_label=baseline_exp_name,
            curriculum_exp_id=curriculum_exp_id,
            curriculum_exp_label=curriculum_exp_name,
            metric_name=comparison_spec["metric"],
            y_label=comparison_spec["y_label"],
            file_name=comparison_spec["file_name"],
            baseline_color=comparison_spec["baseline_color"],
            curriculum_color=comparison_spec["curriculum_color"],
            baseline_label=COMPARISON_BASELINE_LABEL,
            curriculum_label=COMPARISON_CURRICULUM_LABEL,
            clip_range=comparison_spec.get("clip"),
            y_ticks=comparison_spec.get("yticks"),
        )
