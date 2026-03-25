# -*- coding: utf-8 -*-
"""
实验指标可视化对比代码
功能：通过SwanLab OpenApi获取occt项目实验数据，绘制关键训练曲线
适用场景：硕士毕业设计中实验结果的量化分析与可视化展示
"""

from plt_cn_utils import *
from swanlab import OpenApi
import os
import numpy as np
import matplotlib.pyplot as plt


CUT_ITER = 300
EMA_ALPHA = 0.15
FLUCTUATION_ALPHA = 0.20
FLUCTUATION_SCALE = 1.35
DRAW_RAW_CURVE = False

STEP_REWARD_METRICS = {
    "total": "train_info_reward_total",
    "base": [
        "train_info_reward_progress",
        "train_info_reward_vel",
        "train_info_reward_goal",
        "train_info_penalty_near_boundary",
        "train_info_penalty_near_other_agents",
        "train_info_penalty_change_steering",
        "train_info_penalty_change_acc",
        "train_info_penalty_collide_with_agents",
        "train_info_penalty_outside_boundaries",
        "train_info_penalty_backward",
    ],
    "hinge": [
        "train_info_reward_hinge",
        "train_info_reward_hinge_vel",
        "train_info_reward_hinge_ref",
        "train_info_reward_hinge_space",
        "train_info_reward_approach_hinge",
        "train_info_penalty_hinge_time_cost",
    ],
    "platoon": [
        "train_info_reward_platoon_heading",
        "train_info_reward_platoon_vel",
        "train_info_reward_platoon_ref",
        "train_info_reward_platoon_space",
    ],
}


def _fetch_metrics(swanlab_api: OpenApi, exp_cuid: str, metric_keys: list[str]):
    unique_metric_keys = list(dict.fromkeys(metric_keys))
    if not unique_metric_keys:
        raise ValueError("没有可获取的指标字段。")
    return swanlab_api.get_metrics(exp_id=exp_cuid, keys=unique_metric_keys).data


def _ema_smooth(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    if values.size == 0:
        return values

    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * values[idx] + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def _compute_fluctuation_bounds(
    raw_values: np.ndarray,
    center_values: np.ndarray,
    alpha: float = FLUCTUATION_ALPHA,
    scale: float = FLUCTUATION_SCALE,
    clip_range: tuple[float, float] | None = None,
):
    deviation = np.abs(raw_values - center_values)
    band_width = _ema_smooth(deviation, alpha=alpha) * scale
    lower_values = center_values - band_width
    upper_values = center_values + band_width

    if clip_range is not None:
        lower_values = np.clip(lower_values, clip_range[0], clip_range[1])
        upper_values = np.clip(upper_values, clip_range[0], clip_range[1])

    return lower_values, upper_values


def _prepare_aligned_frame(metric_data, metric_keys: list[str]):
    aligned_frame = metric_data[metric_keys].iloc[:CUT_ITER].copy()
    aligned_frame = aligned_frame.dropna(how="any")
    if aligned_frame.empty:
        raise ValueError(f"指标数据为空：{metric_keys}")
    return aligned_frame


def _prepare_series(series):
    sliced_series = series.iloc[:CUT_ITER].dropna()
    if sliced_series.empty:
        raise ValueError(f"指标数据为空：{series.name}")
    return (
        sliced_series.index.to_numpy(),
        sliced_series.to_numpy(dtype=float),
    )


def _sum_metric_series(metric_data, metric_keys: list[str]):
    if not metric_keys:
        raise ValueError("没有可聚合的指标字段。")
    component_frame = metric_data[metric_keys].iloc[:CUT_ITER].copy()
    aggregated_series = component_frame.sum(axis=1, min_count=1)
    aggregated_series.name = "+".join(metric_keys)
    return aggregated_series


def _style_axes(ax, y_label: str, show_legend: bool = False):
    ax.set_ylabel(y_label, fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax.set_xlabel("训练步数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax.tick_params(
        axis="both",
        labelsize=font_size_tick,
        pad=1,
        direction="in",
        top=False,
        right=False,
        labelfontfamily="Times New Roman",
    )
    ax.margins(x=0)
    if show_legend:
        ax.legend(
            loc="best",
            fontsize=font_size_legend,
            prop=font_prop_chinese,
            handlelength=1.8,
            borderpad=0.2,
            labelspacing=0.2,
        )


def _save_figure(save_dir: str, file_name: str):
    plt.savefig(
        os.path.join(save_dir, file_name),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    plt.close()


def plot_fill_style(
    swanlab_api: OpenApi,
    exp_cuid: str,
    metric_keys: dict,
    file_name: str,
    y_label: str,
    color: str,
    save_dir: str,
    mean_label: str,
    band_label: str,
):
    center_key = metric_keys["center"]
    upper_key = metric_keys["upper"]
    lower_key = metric_keys["lower"]
    metric_data = _fetch_metrics(
        swanlab_api, exp_cuid, [center_key, upper_key, lower_key]
    )
    aligned_frame = _prepare_aligned_frame(metric_data, [center_key, upper_key, lower_key])

    x_axis = aligned_frame.index.to_numpy()
    center_values = aligned_frame[center_key].to_numpy(dtype=float)
    upper_values = aligned_frame[upper_key].to_numpy(dtype=float)
    lower_values = aligned_frame[lower_key].to_numpy(dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    ax.plot(
        x_axis,
        center_values,
        label=mean_label,
        linewidth=1.2,
        color=color,
        alpha=0.9,
    )
    ax.plot(x_axis, upper_values, linewidth=0.6, color=color, alpha=0.4)
    ax.plot(x_axis, lower_values, linewidth=0.6, color=color, alpha=0.4)
    ax.fill_between(
        x_axis,
        lower_values,
        upper_values,
        alpha=0.2,
        color=color,
        label=band_label,
    )

    _style_axes(ax, y_label, show_legend=True)
    _save_figure(save_dir, file_name)


def plot_multi_fill_style(
    swanlab_api: OpenApi,
    exp_cuid: str,
    series_specs: list[dict],
    file_name: str,
    y_label: str,
    save_dir: str,
):
    metric_keys = []
    for spec in series_specs:
        metric_keys.extend(spec["center"])
        metric_keys.extend(spec["upper"])
        metric_keys.extend(spec["lower"])

    metric_data = _fetch_metrics(swanlab_api, exp_cuid, metric_keys)
    aligned_frame = _prepare_aligned_frame(metric_data, list(dict.fromkeys(metric_keys)))
    x_axis = aligned_frame.index.to_numpy()

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)

    for spec in series_specs:
        center_values = aligned_frame[spec["center"]].mean(axis=1).to_numpy(dtype=float)
        upper_values = aligned_frame[spec["upper"]].max(axis=1).to_numpy(dtype=float)
        lower_values = aligned_frame[spec["lower"]].min(axis=1).to_numpy(dtype=float)

        ax.plot(
            x_axis,
            center_values,
            label=spec["label"],
            linewidth=1.2,
            color=spec["color"],
            alpha=0.95,
        )
        ax.plot(
            x_axis,
            upper_values,
            linewidth=0.5,
            color=spec["color"],
            alpha=0.35,
        )
        ax.plot(
            x_axis,
            lower_values,
            linewidth=0.5,
            color=spec["color"],
            alpha=0.35,
        )
        ax.fill_between(
            x_axis,
            lower_values,
            upper_values,
            alpha=0.14,
            color=spec["color"],
        )

    _style_axes(ax, y_label, show_legend=True)
    _save_figure(save_dir, file_name)


def plot_ema_style(
    series_specs: list[dict],
    file_name: str,
    y_label: str,
    save_dir: str,
    draw_raw_curve: bool = DRAW_RAW_CURVE,
):
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)

    for spec in series_specs:
        x_axis, raw_values = _prepare_series(spec["series"])
        smooth_values = _ema_smooth(raw_values)
        lower_values, upper_values = _compute_fluctuation_bounds(
            raw_values,
            smooth_values,
            clip_range=spec.get("clip"),
        )

        if draw_raw_curve:
            ax.plot(
                x_axis,
                raw_values,
                linewidth=0.7,
                color=spec["color"],
                alpha=0.16,
            )
        ax.plot(x_axis, upper_values, linewidth=0.5, color=spec["color"], alpha=0.35)
        ax.plot(x_axis, lower_values, linewidth=0.5, color=spec["color"], alpha=0.35)
        ax.fill_between(
            x_axis,
            lower_values,
            upper_values,
            color=spec["color"],
            alpha=0.14,
        )
        ax.plot(
            x_axis,
            smooth_values,
            label=spec["label"],
            linewidth=1.2,
            color=spec["color"],
            alpha=0.95,
        )

    _style_axes(ax, y_label, show_legend=True)
    _save_figure(save_dir, file_name)


def plot_episode_reward(
    swanlab_api: OpenApi,
    exp_cuid: str,
    save_dir: str,
):
    plot_fill_style(
        swanlab_api=swanlab_api,
        exp_cuid=exp_cuid,
        metric_keys={
            "center": "train_reward_episode_reward_mean",
            "upper": "train_reward_episode_reward_max",
            "lower": "train_reward_episode_reward_min",
        },
        file_name="train_reward_episode.pdf",
        y_label="单轮奖励值",
        color="#6d63ff",
        save_dir=save_dir,
        mean_label="单轮奖励均值",
        band_label="单轮奖励区间",
    )


def plot_step_reward_components(
    swanlab_api: OpenApi,
    exp_cuid: str,
    save_dir: str,
):
    all_metric_keys = (
        [STEP_REWARD_METRICS["total"]]
        + STEP_REWARD_METRICS["base"]
        + STEP_REWARD_METRICS["hinge"]
        + STEP_REWARD_METRICS["platoon"]
    )
    metric_data = _fetch_metrics(swanlab_api, exp_cuid, all_metric_keys)

    base_series = _sum_metric_series(metric_data, STEP_REWARD_METRICS["base"])
    hinge_series = _sum_metric_series(metric_data, STEP_REWARD_METRICS["hinge"])
    platoon_series = _sum_metric_series(metric_data, STEP_REWARD_METRICS["platoon"])

    series_specs = [
        {
            "label": "总单步奖励",
            "series": metric_data[STEP_REWARD_METRICS["total"]],
            "color": "#8c564b",
        },
        {
            "label": "基础奖励",
            "series": base_series,
            "color": "#7f7f7f",
        },
        {
            "label": "铰接奖励",
            "series": hinge_series,
            "color": "#17becf",
        },
        {
            "label": "编队奖励",
            "series": platoon_series,
            "color": "#e377c2",
        },
    ]
    plot_ema_style(
        series_specs=series_specs,
        file_name="train_reward_step_components.pdf",
        y_label="单步奖励值",
        save_dir=save_dir,
    )


def plot_probability_curves(
    swanlab_api: OpenApi,
    exp_cuid: str,
    save_dir: str,
):
    probability_specs = [
        {
            "metric": "train_info_done_all_hinged_rate",
            "label": "成功率",
            "color": "#1f77b4",
            "clip": (0.0, 1.0),
        },
        {
            "metric": "train_info_done_collision_with_agents_rate",
            "label": "车辆碰撞",
            "color": "#d62728",
            "clip": (0.0, 1.0),
        },
        {
            "metric": "train_info_done_collision_with_lanelets_rate",
            "label": "越界碰撞",
            "color": "#ff7f0e",
            "clip": (0.0, 1.0),
        },
        {
            "metric": "train_info_done_collision_with_exit_segments_rate",
            "label": "超时未铰接",
            "color": "#2ca02c",
            "clip": (0.0, 1.0),
        },
    ]
    metric_data = _fetch_metrics(
        swanlab_api, exp_cuid, [spec["metric"] for spec in probability_specs]
    )

    series_specs = [
        {
            "label": spec["label"],
            "series": metric_data[spec["metric"]],
            "color": spec["color"],
            "clip": spec.get("clip"),
        }
        for spec in probability_specs
    ]
    plot_ema_style(
        series_specs=series_specs,
        file_name="train_probability_rates.pdf",
        y_label="概率",
        save_dir=save_dir,
    )


def plot_step_length(
    swanlab_api: OpenApi,
    exp_cuid: str,
    save_dir: str,
):
    plot_multi_fill_style(
        swanlab_api=swanlab_api,
        exp_cuid=exp_cuid,
        series_specs=[
            {
                "label": "环岛",
                "center": [
                    "train_road_0_mean_step",
                    "train_road_1_mean_step",
                ],
                "upper": [
                    "train_road_0_max_step",
                    "train_road_1_max_step",
                ],
                "lower": [
                    "train_road_0_min_step",
                    "train_road_1_min_step",
                ],
                "color": "#c187ff",
            },
            {
                "label": "直角弯",
                "center": [
                    "train_road_2_mean_step",
                    "train_road_3_mean_step",
                ],
                "upper": [
                    "train_road_2_max_step",
                    "train_road_3_max_step",
                ],
                "lower": [
                    "train_road_2_min_step",
                    "train_road_3_min_step",
                ],
                "color": "#ffae83",
            },
            {
                "label": "S弯",
                "center": [
                    "train_road_4_mean_step",
                    "train_road_5_mean_step",
                ],
                "upper": [
                    "train_road_4_max_step",
                    "train_road_5_max_step",
                ],
                "lower": [
                    "train_road_4_min_step",
                    "train_road_5_min_step",
                ],
                "color": "#52eec0",
            },
        ],
        file_name="train_road_step.pdf",
        y_label="步长",
        save_dir=save_dir,
    )


if __name__ == "__main__":
    save_dir = "./outputs/occt_vis/"
    experiment_id = "jx4yxkwut28cb8i2cl62h"
    os.makedirs(save_dir, exist_ok=True)

    swanlab_api = OpenApi()
    plot_episode_reward(swanlab_api, experiment_id, save_dir)
    # plot_step_reward_components(swanlab_api, experiment_id, save_dir)
    # plot_probability_curves(swanlab_api, experiment_id, save_dir)
    # plot_step_length(swanlab_api, experiment_id, save_dir)
