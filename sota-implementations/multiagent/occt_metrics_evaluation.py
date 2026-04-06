import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
try:
    import yaml
except ImportError:
    yaml = None

from plt_cn_utils import *


DEFAULT_DT = 0.05
DEFAULT_FOLLOWERS = [1, 2, 3]
DEFAULT_SEDAN_MASS_KG = 1500.0
DEFAULT_SEDAN_WIDTH_M = 1.5
DEFAULT_SEDAN_HEIGHT_M = 1.45
DEFAULT_DRAG_COEFF = 0.29
DEFAULT_ROLLING_RESISTANCE = 0.01
DEFAULT_AIR_DENSITY_KG_PER_M3 = 1.225
DEFAULT_DRIVETRAIN_EFFICIENCY = 0.90
DEFAULT_JERK_TIME_SCALE_SEC = 0.5
DEFAULT_JERK_PENALTY_WEIGHT = 0.10
GRAVITY_M_PER_S2 = 9.81
DEFAULT_VALIDATION_RESULT_DIR = (
    Path(__file__).resolve().parents[3]
    / "VMAS_occt"
    / "vmas"
    / "scenarios"
    / "occt_scenario_test_result"
)
DEFAULT_PLOT_DIR = (
    Path(__file__).resolve().parents[2] / "outputs" / "occt_vis"
)
ROAD_TYPE_BY_ID = {
    0: "roundabout",
    1: "roundabout",
    2: "right_angle_turn",
    3: "right_angle_turn",
    4: "s_curve",
    5: "s_curve",
}
METHOD_PLOT_ORDER = [
    "marl_baseline",
    "marl_full_obse",
    "marl_lipsnet++",
    "marl_mlp_continuous_w_cp_his10",
    "marl_lipsnet_continuous_his10",
    "marl_lipsnet_phase_blending_his10",
    "marl",
    "pid",
    "mppi",
]
METHOD_COLORS = {
    "marl_baseline": "#1f77b4",
    "marl_full_obse": "#ff7f0e",
    "marl_lipsnet++": "#2ca02c",
    "marl_mlp_continuous_w_cp_his10": "#d62728",
    "marl_mlp_continues_w_cp_his10": "#d62728",
    "marl_lipsnet_continuous_his10": "#9467bd",
    "marl_lipsnet_phase_blending_his10": "#8c564b",
    "marl": "#1f77b4",
    "pid": "#ff7f0e",
    "mppi": "#2ca02c",
}
METHOD_DISPLAY_NAMES = {
    "marl_baseline": "MARL Baseline",
    "marl_full_obse": "MARL Full Obse",
    "marl_lipsnet++": "MARL LipsNet++",
    "marl_mlp_continuous_w_cp_his10": "MLP Continuous Act Penalty (H=10)",
    "marl_mlp_continues_w_cp_his10": "MLP Continuous Act Penalty (H=10)",
    "marl_lipsnet_continuous_his10": "LipsNet Continue Baseline (H=10)",
    "marl_lipsnet_phase_blending_his10": "LipsNet Phase Blending (H=10)",
    "marl": "MARL",
    "pid": "PID",
    "mppi": "MPPI",
}
METHOD_NAME_ALIASES = {
    "marl_mlp_continues_w_cp_his10": "marl_mlp_continuous_w_cp_his10",
}
AGENT_COLORS = {
    0: "#7f7f7f",
    1: "#9467bd",
    2: "#8c564b",
    3: "#17becf",
    4: "#e377c2",
    5: "#bcbd22",
}
# PLOT_METHOD_LINE_PALETTES = {
#     "longitudinal_error": ["#5e3c99", "#8073ac", "#b2abd2"],
#     "lateral_error": ["#8c510a", "#bf812d", "#dfc27d"],
#     "hinge_dis": ["#01665e", "#35978f", "#80cdc1"],
#     "hinge_status": ["#4d9221", "#7fbc41", "#b8e186"],
#     "speed": ["#08519c", "#3182bd", "#6baed6"],
#     "acceleration": ["#980043", "#dd1c77", "#df65b0"],
#     "steering_angle": ["#252525", "#636363", "#969696"],
#     "heading_angle": ["#006d2c", "#31a354", "#74c476"],
# }
PLOT_METHOD_LINE_PALETTES = {
    # Viridis 色系（青→黄→绿，顶级区分度）
    "longitudinal_error": ["#00204d", "#4d7399", "#ffce51"],
    
    # Plasma 色系（紫→粉→黄，高对比）
    "lateral_error": ["#3e0466", "#d9576e", "#f9e826"],
    
    # Inferno 色系（黑→红→黄）
    "hinge_dis": ["#A519B1", "#bb3654", "#ff8f1f"],
    
    # Cividis 色系（蓝→黄，色盲最优）
    "hinge_status": ["#440154", "#2a788e", "#7ad151"],
    
    # Blues → 科学色（深蓝→青）
    "speed": ["#08306b", "#2c81c9", "#9ac9e3"],
    
    # Rocket 色系（深棕→红）
    "acceleration": ["#FD2D2D", "#f7af43", "#add929"],
    
    # Greys 专业灰度
    "steering_angle": ["#0dcd80", "#1FA0B6", "#1632bb"],
    
    # Turbid 清爽蓝绿
    "heading_angle": ["#2d5a70", "#41b399", "#9fd799"],
}


def _first_done_index(traj) -> Optional[int]:
    next_done = traj.get(("next", "done")).sum(
        tuple(range(traj.batch_dims, traj.get(("next", "done")).ndim)),
        dtype=torch.bool,
    )
    if next_done.any():
        return int(next_done.nonzero(as_tuple=True)[0][0].item())
    return None


def get_valid_length(traj) -> int:
    first_done = _first_done_index(traj)
    if first_done is None:
        return int(traj.batch_size[0])
    return first_done + 1


def _safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def _safe_std(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float32).std(unbiased=False).item())


def _safe_min(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(min(values))


def _safe_max(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(max(values))


def _normalize_method_name(name: str) -> str:
    normalized = name.strip().lower()
    return METHOD_NAME_ALIASES.get(normalized, normalized)


def _get_method_display_name(method: str) -> str:
    if method in METHOD_DISPLAY_NAMES:
        return METHOD_DISPLAY_NAMES[method]
    return method.replace("_", " ")


def _infer_rollout_method_name(input_path: Path, label: Optional[str] = None) -> str:
    if label:
        return _normalize_method_name(label)

    path_text = input_path.as_posix().lower()
    if (
        "phase_blending_his10" in path_text
        or "phase-blending-his10" in path_text
        or "phase_blending" in path_text
    ):
        return "marl_lipsnet_phase_blending_his10"
    if (
        "lipsnet_continuous_his10" in path_text
        or "lipsnet-continuous-his10" in path_text
        or "continuous_his10" in path_text
    ):
        return "marl_lipsnet_continuous_his10"
    if (
        "mlp_continues_w_cp_his10" in path_text
        or "mlp-continues-w-cp-his10" in path_text
        or "continues_w_cp_his10" in path_text
    ):
        return "marl_mlp_continues_w_cp_his10"
    if "lipsnet++" in path_text or "lipsnetpp" in path_text or "lipsnet" in path_text:
        return "marl_lipsnet++"
    if (
        "full_obse" in path_text
        or "full-obse" in path_text
        or "full_observation" in path_text
        or "fullobse" in path_text
    ):
        return "marl_full_obse"
    if "baseline" in path_text:
        return "marl_baseline"
    return "marl"


def _resolve_method_name(requested: str, available_methods: Sequence[str]) -> str:
    normalized = _normalize_method_name(requested)
    available = list(dict.fromkeys(_normalize_method_name(method) for method in available_methods))
    if normalized in available:
        return normalized

    prefix_matches = [method for method in available if method.startswith(f"{normalized}_")]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    if len(prefix_matches) > 1:
        raise ValueError(
            f"Method '{requested}' is ambiguous. Matching methods: {sorted(prefix_matches)}"
        )
    raise ValueError(
        f"Method '{requested}' not found in loaded inputs. Available methods: {sorted(available)}"
    )


def _str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value '{value}'. Use true or false."
    )


def _squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    tensor = value.float() if value.dtype.is_floating_point else value
    if tensor.ndim > 0 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


def _get_road_type(road_id: Optional[int]) -> str:
    if road_id is None:
        return "unknown"
    return ROAD_TYPE_BY_ID.get(int(road_id), "other")


def _latest_true_segment_until(
    mask: torch.Tensor, end_index: int
) -> tuple[Optional[int], Optional[int]]:
    if end_index < 0:
        return None, None
    end_index = min(end_index, mask.shape[0] - 1)
    true_indices = torch.nonzero(mask[: end_index + 1], as_tuple=True)[0]
    if len(true_indices) == 0:
        return None, None

    segment_end = int(true_indices[-1].item())
    segment_start = segment_end
    while segment_start > 0 and bool(mask[segment_start - 1].item()):
        segment_start -= 1
    return segment_start, segment_end


def _default_road_name(road_id: Optional[int]) -> str:
    if road_id is None:
        return "unknown"
    return f"road_{int(road_id)}"


def _extract_episode_road_id(
    episode: Dict[str, Any], fallback_road_id: Optional[int] = None
) -> Optional[int]:
    if "road_id" in episode and episode["road_id"] is not None:
        return int(episode["road_id"])
    info = episode.get("info")
    if isinstance(info, dict) and "road_batch_id" in info:
        road_tensor = _squeeze_trailing_unit_dim(info["road_batch_id"])
        return int(road_tensor.reshape(-1)[-1].item())
    if fallback_road_id is None:
        return None
    return int(fallback_road_id)


def _extract_episode_road_name(
    episode: Dict[str, Any],
    road_id: Optional[int],
    fallback_road_name: Optional[str] = None,
) -> str:
    if episode.get("road_name"):
        return str(episode["road_name"])
    if fallback_road_name:
        return str(fallback_road_name)
    return _default_road_name(road_id)


def _compute_episode_terminal_status(
    episode: Dict[str, Any],
    follower_ids: Optional[Sequence[int]] = None,
) -> Dict[str, bool]:
    info = episode["info"]
    agent_hinge_status = _squeeze_trailing_unit_dim(info["agent_hinge_status"]).bool()
    num_agents = int(agent_hinge_status.shape[1])

    if follower_ids is None:
        resolved_follower_ids = _filter_valid_agent_ids(DEFAULT_FOLLOWERS, num_agents)
    else:
        resolved_follower_ids = _filter_valid_agent_ids(follower_ids, num_agents)

    final_agent_index = 0
    final_collision_agents = bool(
        _squeeze_trailing_unit_dim(info["done_collision_with_agents"])[
            -1, final_agent_index
        ].item()
    )
    final_collision_lanelets = bool(
        _squeeze_trailing_unit_dim(info["done_collision_with_lanelets"])[
            -1, final_agent_index
        ].item()
    )
    final_collision_exit = bool(
        _squeeze_trailing_unit_dim(info["done_collision_with_exit_segments"])[
            -1, final_agent_index
        ].item()
    )
    final_all_hinged = bool(
        agent_hinge_status[-1, resolved_follower_ids].all().item()
    )
    final_success = (
        final_collision_exit
        and final_all_hinged
        and not final_collision_agents
        and not final_collision_lanelets
    )
    return {
        "final_success": final_success,
        "final_collision_agents": final_collision_agents,
        "final_collision_lanelets": final_collision_lanelets,
        "final_collision_exit": final_collision_exit,
        "final_all_hinged": final_all_hinged,
    }


def _count_hinge_opportunity_successes(
    hinge_status: torch.Tensor,
    agent_hinge_status: torch.Tensor,
) -> tuple[int, int]:
    hinge_status = _squeeze_trailing_unit_dim(hinge_status).bool()
    agent_hinge_status = _squeeze_trailing_unit_dim(agent_hinge_status).bool()
    if hinge_status.shape != agent_hinge_status.shape:
        raise ValueError(
            "hinge_status and agent_hinge_status must share the same shape when "
            "counting hinge opportunity successes: "
            f"{tuple(hinge_status.shape)} vs {tuple(agent_hinge_status.shape)}"
        )

    success_count = 0
    opportunity_count = 0
    num_followers = int(hinge_status.shape[1])

    for follower_idx in range(num_followers):
        for _, _, success_index in _iter_hinge_opportunities(
            hinge_status[:, follower_idx],
            agent_hinge_status[:, follower_idx],
        ):
            opportunity_count += 1
            if success_index is not None:
                success_count += 1

    return success_count, opportunity_count


def _iter_hinge_opportunities(
    hinge_status_series: torch.Tensor,
    agent_hinged_series: torch.Tensor,
):
    hinge_status_series = _squeeze_trailing_unit_dim(hinge_status_series).bool()
    agent_hinged_series = _squeeze_trailing_unit_dim(agent_hinged_series).bool()
    if hinge_status_series.shape != agent_hinged_series.shape:
        raise ValueError(
            "hinge_status_series and agent_hinged_series must share the same shape "
            f"when iterating hinge opportunities: {tuple(hinge_status_series.shape)} "
            f"vs {tuple(agent_hinged_series.shape)}"
        )

    time_horizon = int(hinge_status_series.shape[0])
    step = 0
    while step < time_horizon:
        if not bool(hinge_status_series[step].item()):
            step += 1
            continue
        segment_start = step
        while step + 1 < time_horizon and bool(hinge_status_series[step + 1].item()):
            step += 1
        segment_end = step
        success_indices = torch.nonzero(
            agent_hinged_series[segment_start : segment_end + 1],
            as_tuple=True,
        )[0]
        success_index = (
            segment_start + int(success_indices[0].item())
            if len(success_indices) > 0
            else None
        )
        yield segment_start, segment_end, success_index
        step += 1


def _build_result_bundle(
    *,
    method: str,
    episodes: List[Dict[str, Any]],
    dt: float,
    followers: List[int],
    source_format: str,
    road_name: Optional[str] = None,
    requested_road_id: Optional[int] = None,
    report_type: Optional[str] = None,
    road_type: Optional[str] = None,
    road_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    bundle = {
        "format": "occt_aggregated_report",
        "source_format": source_format,
        "method": method,
        "road_name": road_name,
        "requested_road_id": requested_road_id,
        "episodes_requested": len(episodes),
        "episodes_completed": len(episodes),
        "dt": dt,
        "followers": followers,
        "episodes": episodes,
    }
    if report_type is not None:
        bundle["report_type"] = report_type
    if road_type is not None:
        bundle["road_type"] = road_type
    if road_ids is not None:
        bundle["road_ids"] = list(road_ids)
    return bundle


def build_result_data_from_rollout(
    rollouts,
    *,
    followers: List[int],
    dt: float,
    method: str = "marl",
) -> Dict[str, Any]:
    trajectories = list(rollouts.unbind(0))
    episodes = []
    road_ids = []
    for episode_index, traj in enumerate(trajectories):
        valid_len = get_valid_length(traj)
        info = {
            key: value[:valid_len].detach().cpu()
            for key, value in traj["next"]["agents"]["info"].items()
        }
        road_id = None
        if "road_batch_id" in info:
            road_id = int(_squeeze_trailing_unit_dim(info["road_batch_id"]).reshape(-1)[-1].item())
            road_ids.append(road_id)
        episodes.append(
            {
                "episode_index": episode_index,
                "num_steps": valid_len,
                "info": info,
                "road_id": road_id,
                "road_name": _default_road_name(road_id),
            }
        )
    unique_road_ids = sorted(set(road_ids))
    return _build_result_bundle(
        method=method,
        episodes=episodes,
        dt=dt,
        followers=followers,
        source_format="rollout",
        road_name=None,
        requested_road_id=None,
        road_ids=unique_road_ids,
    )


def _resolve_followers(
    result_data: Dict[str, Any],
    followers: Optional[List[int]],
) -> List[int]:
    if followers is not None:
        return followers
    metadata_followers = result_data.get("followers")
    if metadata_followers:
        return [int(agent_id) for agent_id in metadata_followers]
    return list(DEFAULT_FOLLOWERS)


def _angle_diff_deg(rot: torch.Tensor, target_vel: torch.Tensor) -> Optional[float]:
    rot = torch.as_tensor(rot, dtype=torch.float32).reshape(-1)
    if rot.numel() == 0:
        return None
    rot = rot[0]

    target_vel = torch.as_tensor(target_vel, dtype=torch.float32).reshape(-1)
    if target_vel.numel() < 2:
        return None
    target_vel = target_vel[:2]

    target_speed = torch.linalg.norm(target_vel).item()
    if target_speed <= 1e-6:
        return None
    agent_heading = torch.stack((torch.cos(rot), torch.sin(rot)))
    target_heading = target_vel / target_vel.norm().clamp_min(1e-6)
    cross = agent_heading[0] * target_heading[1] - agent_heading[1] * target_heading[0]
    dot = (agent_heading * target_heading).sum().clamp(-1.0, 1.0)
    angle_diff = torch.atan2(torch.abs(cross), dot) * (180.0 / torch.pi)
    return float(angle_diff.item())


def _compute_signal_rate(signal: torch.Tensor, dt: float) -> torch.Tensor:
    series = torch.as_tensor(signal, dtype=torch.float32).reshape(-1)
    if series.numel() == 0:
        return series
    rate = torch.zeros_like(series)
    if series.numel() > 1:
        rate[1:] = (series[1:] - series[:-1]) / max(dt, 1e-6)
    return rate


def _compute_positive_specific_energy_proxy(
    speed_series: torch.Tensor,
    acceleration_series: torch.Tensor,
    dt: float,
) -> float:
    speed_series = torch.as_tensor(speed_series, dtype=torch.float32).reshape(-1)
    acceleration_series = torch.as_tensor(
        acceleration_series, dtype=torch.float32
    ).reshape(-1)
    if speed_series.numel() == 0 or acceleration_series.numel() == 0:
        return 0.0
    if speed_series.shape != acceleration_series.shape:
        raise ValueError(
            "speed_series and acceleration_series must have the same shape for "
            "energy proxy computation: "
            f"{tuple(speed_series.shape)} vs {tuple(acceleration_series.shape)}"
        )
    # Positive specific traction power proxy: max(v * a, 0), integrated over time.
    # Unit: J/kg.
    positive_specific_power = torch.clamp(speed_series * acceleration_series, min=0.0)
    specific_energy_j_per_kg = positive_specific_power.sum() * max(dt, 1e-6)
    return float(specific_energy_j_per_kg.item())


def _compute_jerk_augmented_energy_proxy(
    speed_series: torch.Tensor,
    acceleration_series: torch.Tensor,
    jerk_series: torch.Tensor,
    dt: float,
    *,
    vehicle_mass_kg: float = DEFAULT_SEDAN_MASS_KG,
    vehicle_width_m: float = DEFAULT_SEDAN_WIDTH_M,
    vehicle_height_m: float = DEFAULT_SEDAN_HEIGHT_M,
    drag_coeff: float = DEFAULT_DRAG_COEFF,
    rolling_resistance_coeff: float = DEFAULT_ROLLING_RESISTANCE,
    air_density_kg_per_m3: float = DEFAULT_AIR_DENSITY_KG_PER_M3,
    drivetrain_efficiency: float = DEFAULT_DRIVETRAIN_EFFICIENCY,
    jerk_time_scale_sec: float = DEFAULT_JERK_TIME_SCALE_SEC,
    jerk_penalty_weight: float = DEFAULT_JERK_PENALTY_WEIGHT,
) -> float:
    speed_series = torch.as_tensor(speed_series, dtype=torch.float32).reshape(-1)
    acceleration_series = torch.as_tensor(
        acceleration_series, dtype=torch.float32
    ).reshape(-1)
    jerk_series = torch.as_tensor(jerk_series, dtype=torch.float32).reshape(-1)
    if (
        speed_series.numel() == 0
        or acceleration_series.numel() == 0
        or jerk_series.numel() == 0
    ):
        return 0.0
    if not (
        speed_series.shape == acceleration_series.shape == jerk_series.shape
    ):
        raise ValueError(
            "speed_series, acceleration_series, and jerk_series must have the same "
            "shape for jerk-augmented energy proxy computation: "
            f"{tuple(speed_series.shape)}, {tuple(acceleration_series.shape)}, "
            f"{tuple(jerk_series.shape)}"
        )

    speed_series = torch.clamp(speed_series, min=0.0)
    frontal_area_m2 = vehicle_width_m * vehicle_height_m
    positive_inertial_power = vehicle_mass_kg * torch.clamp(
        acceleration_series, min=0.0
    ) * speed_series
    rolling_power = (
        rolling_resistance_coeff * vehicle_mass_kg * GRAVITY_M_PER_S2 * speed_series
    )
    aerodynamic_power = (
        0.5
        * air_density_kg_per_m3
        * drag_coeff
        * frontal_area_m2
        * speed_series**3
    )
    jerk_penalty_power = (
        jerk_penalty_weight
        * vehicle_mass_kg
        * jerk_time_scale_sec
        * torch.abs(jerk_series)
        * speed_series
    )
    total_power = (
        positive_inertial_power
        + rolling_power
        + aerodynamic_power
        + jerk_penalty_power
    ) / max(drivetrain_efficiency, 1e-6)
    energy_wh = total_power.sum() * max(dt, 1e-6) / 3600.0
    return float(energy_wh.item())


def _compute_yaw_rate_from_rot(rot: torch.Tensor, dt: float) -> torch.Tensor:
    rot_series = torch.as_tensor(rot, dtype=torch.float32).reshape(-1)
    if rot_series.numel() == 0:
        return rot_series
    yaw_rate = torch.zeros_like(rot_series)
    if rot_series.numel() > 1:
        angle_delta = torch.atan2(
            torch.sin(rot_series[1:] - rot_series[:-1]),
            torch.cos(rot_series[1:] - rot_series[:-1]),
        )
        yaw_rate[1:] = angle_delta * (180.0 / torch.pi) / max(dt, 1e-6)
    return yaw_rate


def compute_rollout_metrics_from_object(
    rollouts,
    *,
    followers: List[int],
    dt: float,
) -> Dict[str, object]:
    result_data = build_result_data_from_rollout(
        rollouts,
        followers=followers,
        dt=dt,
    )
    return compute_validation_metrics_from_object(
        result_data,
        followers=followers,
        dt=dt,
    )


def compute_rollout_metrics(
    rollout_path: str,
    *,
    followers: List[int],
    dt: float,
) -> Dict[str, object]:
    rollouts = torch.load(rollout_path, map_location="cpu", weights_only=False)
    return compute_rollout_metrics_from_object(rollouts, followers=followers, dt=dt)


def compute_validation_metrics_from_object(
    result_data: Dict[str, Any],
    *,
    followers: Optional[List[int]],
    dt: Optional[float],
) -> Dict[str, object]:
    episodes = result_data.get("episodes", [])
    resolved_followers = _resolve_followers(result_data, followers)
    resolved_dt = float(result_data.get("dt", DEFAULT_DT) if dt is None else dt)

    platoon_abs_all = []
    platoon_front = []
    platoon_back = []
    platoon_lateral_tracking_errors = []
    platoon_ttc_episode_mins = []
    control_acc_abs_values = []
    control_jerk_abs_values = []
    control_steering_rate_abs_values = []

    hinge_times = []
    hinge_speed_diffs = []
    hinge_pre_dock_vel_angle_diffs = []
    hinge_gate_angle_diffs = []
    final_hinged_counts = []
    final_hinged_ratios = []
    agent_hinged_time_values = []
    agent_hinged_time_ratio_values = []
    energy_proxy_values = []
    energy_proxy_jerk_values = []
    energy_proxy_rate_values = []
    energy_proxy_jerk_rate_values = []

    hinge_times_by_agent = {agent: [] for agent in resolved_followers}
    hinge_speed_diffs_by_agent = {agent: [] for agent in resolved_followers}
    hinge_pre_dock_vel_angle_diffs_by_agent = {agent: [] for agent in resolved_followers}
    hinge_gate_angle_diffs_by_agent = {agent: [] for agent in resolved_followers}

    collision_flags = []
    collision_agents_flags = []
    collision_lanelets_flags = []
    collision_exit_flags = []
    terminal_success_flags = []
    hinge_opportunity_success_count = 0
    hinge_opportunity_total_count = 0
    road_ids = []

    episode_lengths = []

    for episode in episodes:
        info = episode["info"]
        episode_lengths.append(int(episode["num_steps"]))

        error_space = info["error_space"][:, resolved_followers, :].float()
        hinge_status = _squeeze_trailing_unit_dim(
            info["hinge_status"][:, resolved_followers]
        ).bool()
        agent_hinge_status = _squeeze_trailing_unit_dim(
            info["agent_hinge_status"][:, resolved_followers]
        ).bool()
        hinge_steps = _squeeze_trailing_unit_dim(
            info["hinge_steps"][:, resolved_followers]
        ).float()
        s_all = _squeeze_trailing_unit_dim(info["s"]).float()
        vel_all = info["vel"].float()
        speed_all = torch.linalg.norm(vel_all, dim=-1)
        distance_ref = _squeeze_trailing_unit_dim(
            info["distance_ref"][:, resolved_followers]
        ).float()
        vel = info["vel"][:, resolved_followers, :].float()
        hinge_vel = (
            info["hinge_vel"][:, resolved_followers, :].float()
            if "hinge_vel" in info
            else None
        )
        error_vel = (
            info["error_vel"][:, resolved_followers, :].float()
            if "error_vel" in info
            else None
        )
        rot = _squeeze_trailing_unit_dim(info["rot"][:, resolved_followers]).float()
        hinge_gate_angle = (
            _squeeze_trailing_unit_dim(
                info["hinge_gate_angle_diff_deg"][:, resolved_followers]
            ).float()
            if "hinge_gate_angle_diff_deg" in info
            else None
        )
        hinged_steps_per_agent = agent_hinge_status.to(torch.float32).sum(dim=0)
        episode_horizon = max(int(agent_hinge_status.shape[0]), 1)
        agent_hinged_time_values.extend(
            (hinged_steps_per_agent * resolved_dt).detach().cpu().tolist()
        )
        agent_hinged_time_ratio_values.extend(
            (hinged_steps_per_agent / float(episode_horizon)).detach().cpu().tolist()
        )

        for follower_idx, agent_id in enumerate(resolved_followers):
            active_control_mask = ~agent_hinge_status[:, follower_idx]
            actual_acc_series = _compute_signal_rate(speed_all[:, agent_id], resolved_dt)
            actual_jerk_series = _compute_signal_rate(actual_acc_series, resolved_dt)
            if active_control_mask.any():
                active_duration_sec = (
                    float(active_control_mask.to(torch.float32).sum().item())
                    * resolved_dt
                )
                energy_proxy_value = _compute_positive_specific_energy_proxy(
                    speed_all[:, agent_id][active_control_mask],
                    actual_acc_series[active_control_mask],
                    resolved_dt,
                )
                energy_proxy_jerk_value = _compute_jerk_augmented_energy_proxy(
                    speed_all[:, agent_id][active_control_mask],
                    actual_acc_series[active_control_mask],
                    actual_jerk_series[active_control_mask],
                    resolved_dt,
                )
                energy_proxy_values.append(energy_proxy_value)
                energy_proxy_jerk_values.append(energy_proxy_jerk_value)
                if active_duration_sec > 0.0:
                    energy_proxy_rate_values.append(
                        energy_proxy_value / active_duration_sec
                    )
                    energy_proxy_jerk_rate_values.append(
                        energy_proxy_jerk_value / active_duration_sec
                    )

            if not active_control_mask.any():
                continue

            if "act_acc" in info:
                acc_series = _squeeze_trailing_unit_dim(
                    info["act_acc"][:, agent_id]
                ).float().reshape(-1)
            else:
                acc_series = actual_acc_series
            jerk_series = _compute_signal_rate(acc_series, resolved_dt)
            if "steering_rate_abs_deg" in info:
                steering_rate_abs_series = _squeeze_trailing_unit_dim(
                    info["steering_rate_abs_deg"][:, agent_id]
                ).float().reshape(-1)
            else:
                steer_series = _squeeze_trailing_unit_dim(
                    info["act_steer"][:, agent_id]
                ).float().reshape(-1)
                steering_rate_abs_series = (
                    _compute_signal_rate(steer_series, resolved_dt).abs()
                    * (180.0 / torch.pi)
                )

            control_acc_abs_values.extend(
                acc_series[active_control_mask].abs().detach().cpu().tolist()
            )
            control_jerk_abs_values.extend(
                jerk_series[active_control_mask].abs().detach().cpu().tolist()
            )
            control_steering_rate_abs_values.extend(
                steering_rate_abs_series[active_control_mask]
                .abs()
                .detach()
                .cpu()
                .tolist()
            )

        platoon_mask = ~hinge_status
        if platoon_mask.any():
            abs_error = error_space.abs()
            platoon_abs_all.append(float(abs_error[platoon_mask].mean().item()))
            platoon_front.append(float(abs_error[..., 0][platoon_mask].mean().item()))
            platoon_back.append(float(abs_error[..., 1][platoon_mask].mean().item()))
            platoon_lateral_tracking_errors.append(
                float(distance_ref[platoon_mask].mean().item())
            )

        episode_ttc_candidates = []

        for follower_idx, agent_id in enumerate(resolved_followers):
            platoon_step_mask = platoon_mask[:, follower_idx]
            if platoon_step_mask.any():
                ego_speed = speed_all[:, agent_id]
                ego_s = s_all[:, agent_id]

                front_agent_id = agent_id - 1
                if front_agent_id >= 0:
                    front_speed = speed_all[:, front_agent_id]
                    front_s = s_all[:, front_agent_id]
                    front_distance = front_s - ego_s
                    front_closing_speed = ego_speed - front_speed
                    front_ttc_mask = (
                        platoon_step_mask
                        & (front_distance > 0.0)
                        & (front_closing_speed > 0.0)
                    )
                    if front_ttc_mask.any():
                        episode_ttc_candidates.extend(
                            (
                                front_distance[front_ttc_mask]
                                / front_closing_speed[front_ttc_mask].clamp_min(1e-6)
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )

                back_agent_id = agent_id + 1
                if back_agent_id < s_all.shape[1]:
                    back_speed = speed_all[:, back_agent_id]
                    back_s = s_all[:, back_agent_id]
                    back_distance = ego_s - back_s
                    back_closing_speed = back_speed - ego_speed
                    back_ttc_mask = (
                        platoon_step_mask
                        & (back_distance > 0.0)
                        & (back_closing_speed > 0.0)
                    )
                    if back_ttc_mask.any():
                        episode_ttc_candidates.extend(
                            (
                                back_distance[back_ttc_mask]
                                / back_closing_speed[back_ttc_mask].clamp_min(1e-6)
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )

            for hinge_segment_start, _, success_index in _iter_hinge_opportunities(
                hinge_status[:, follower_idx],
                agent_hinge_status[:, follower_idx],
            ):
                if success_index is None:
                    continue

                metric_index = max(success_index - 1, 0)
                hinge_time = float(
                    (success_index - hinge_segment_start + 1) * resolved_dt
                )
                hinge_times.append(hinge_time)
                hinge_times_by_agent[agent_id].append(hinge_time)

                if hinge_vel is not None:
                    speed_diff = float(
                        torch.abs(
                            torch.linalg.norm(vel[metric_index, follower_idx])
                            - torch.linalg.norm(hinge_vel[metric_index, follower_idx])
                        ).item()
                    )
                elif error_vel is not None:
                    speed_diff = float(
                        torch.abs(error_vel[metric_index, follower_idx, 0]).item()
                    )
                else:
                    speed_diff = float("nan")
                hinge_speed_diffs.append(speed_diff)
                hinge_speed_diffs_by_agent[agent_id].append(speed_diff)

                if hinge_vel is not None:
                    pre_dock_vel_angle_diff = _angle_diff_deg(
                        rot[metric_index, follower_idx],
                        hinge_vel[metric_index, follower_idx],
                    )
                    if pre_dock_vel_angle_diff is not None:
                        hinge_pre_dock_vel_angle_diffs.append(pre_dock_vel_angle_diff)
                        hinge_pre_dock_vel_angle_diffs_by_agent[agent_id].append(
                            pre_dock_vel_angle_diff
                        )

                if hinge_gate_angle is not None:
                    gate_angle_diff = float(
                        hinge_gate_angle[success_index, follower_idx].item()
                    )
                    hinge_gate_angle_diffs.append(gate_angle_diff)
                    hinge_gate_angle_diffs_by_agent[agent_id].append(gate_angle_diff)

        if episode_ttc_candidates:
            platoon_ttc_episode_mins.append(float(min(episode_ttc_candidates)))

        terminal_status = _compute_episode_terminal_status(
            episode, follower_ids=resolved_followers
        )
        final_success = terminal_status["final_success"]
        final_collision_agents = terminal_status["final_collision_agents"]
        final_collision_lanelets = terminal_status["final_collision_lanelets"]
        final_collision_exit = terminal_status["final_collision_exit"]
        final_collision_any = (
            final_collision_agents or final_collision_lanelets or final_collision_exit
        )
        final_hinged_count = float(agent_hinge_status[-1].sum().item())
        final_hinged_ratio = (
            final_hinged_count / float(len(resolved_followers))
            if len(resolved_followers) > 0
            else float("nan")
        )
        episode_hinge_success_count, episode_hinge_opportunity_count = (
            _count_hinge_opportunity_successes(
                hinge_status,
                agent_hinge_status,
            )
        )

        terminal_success_flags.append(final_success)
        collision_flags.append(final_collision_any)
        collision_agents_flags.append(final_collision_agents)
        collision_lanelets_flags.append(final_collision_lanelets)
        collision_exit_flags.append(final_collision_exit)
        final_hinged_counts.append(final_hinged_count)
        final_hinged_ratios.append(final_hinged_ratio)
        hinge_opportunity_success_count += episode_hinge_success_count
        hinge_opportunity_total_count += episode_hinge_opportunity_count
        road_id = _extract_episode_road_id(
            episode,
            result_data.get("requested_road_id"),
        )
        if road_id is not None:
            road_ids.append(road_id)

    total_episodes = len(episodes)
    results = {
        "source_format": result_data.get("source_format", "validation"),
        "method": result_data.get("method"),
        "road_name": result_data.get("road_name"),
        "requested_road_id": result_data.get("requested_road_id"),
        "road_ids": sorted(set(road_ids)),
        "episodes": total_episodes,
        "episodes_requested": result_data.get("episodes_requested"),
        "episodes_completed": result_data.get("episodes_completed", total_episodes),
        "episode_length_mean": _safe_mean([float(length) for length in episode_lengths]),
        "episode_length_std": _safe_std([float(length) for length in episode_lengths]),
        "followers": resolved_followers,
        "dt": resolved_dt,
        # "platoon_follow_error_abs_mean": _safe_mean(platoon_abs_all),
        # "platoon_follow_error_abs_std": _safe_std(platoon_abs_all),
        "s_error_mean": _safe_mean(platoon_front),
        "s_error_std": _safe_std(platoon_front),
        "la_error_mean": _safe_mean(
            platoon_lateral_tracking_errors
        ),
        "la_error_std": _safe_std(
            platoon_lateral_tracking_errors
        ),
        # "platoon_follow_back_error_abs_mean": _safe_mean(platoon_back),
        # "platoon_follow_back_error_abs_std": _safe_std(platoon_back),
        "ttc_min_mean": _safe_mean(platoon_ttc_episode_mins),
        "ttc_min_std": _safe_std(platoon_ttc_episode_mins),
        "ttc_global_min": (
            min(platoon_ttc_episode_mins)
            if platoon_ttc_episode_mins
            else float("nan")
        ),
        "ttc_valid_episode_count": len(platoon_ttc_episode_mins),
        "acc_mean": _safe_mean(control_acc_abs_values),
        "acc_min": _safe_min(control_acc_abs_values),
        "acc_max": _safe_max(control_acc_abs_values),
        "acc_std": _safe_std(control_acc_abs_values),
        "jerk_mean": _safe_mean(control_jerk_abs_values),
        "jerk_min": _safe_min(control_jerk_abs_values),
        "jerk_max": _safe_max(control_jerk_abs_values),
        "jerk_std": _safe_std(control_jerk_abs_values),
        "ste_rate_mean": _safe_mean(control_steering_rate_abs_values),
        "ste_rate_min": _safe_min(control_steering_rate_abs_values),
        "ste_rate_max": _safe_max(control_steering_rate_abs_values),
        "ste_rate_std": _safe_std(control_steering_rate_abs_values),
        "hinge_time_mean": _safe_mean(hinge_times),
        "hinge_time_std": _safe_std(hinge_times),
        "hinge_spe_diff_mean": _safe_mean(hinge_speed_diffs),
        "hinge_spe_diff_std": _safe_std(hinge_speed_diffs),
        # "hinge_pre_dock_vel_angle_diff_deg_mean": _safe_mean(
        #     hinge_pre_dock_vel_angle_diffs
        # ),
        # "hinge_pre_dock_vel_angle_diff_deg_std": _safe_std(
        #     hinge_pre_dock_vel_angle_diffs
        # ),
        # "hinge_instant_angle_diff_deg_mean": _safe_mean(
        #     hinge_pre_dock_vel_angle_diffs
        # ),
        # "hinge_instant_angle_diff_deg_std": _safe_std(
        #     hinge_pre_dock_vel_angle_diffs
        # ),
        "hinge_ang_diff_mean": _safe_mean(hinge_gate_angle_diffs),
        "hinge_ang_diff_std": _safe_std(hinge_gate_angle_diffs),
        "hinge_success_event_count": len(hinge_times),
        "hinge_count_mean": _safe_mean(final_hinged_counts),
        "hinge_count_std": _safe_std(final_hinged_counts),
        "hinge_ratio_mean": _safe_mean(final_hinged_ratios),
        "hinge_ratio_std": _safe_std(final_hinged_ratios),
        "agent_hinged_time_mean": _safe_mean(agent_hinged_time_values),
        "agent_hinged_time_std": _safe_std(agent_hinged_time_values),
        "agent_hinged_time_ratio_mean": _safe_mean(agent_hinged_time_ratio_values),
        "agent_hinged_time_ratio_std": _safe_std(agent_hinged_time_ratio_values),
        "energy_proxy_mean": _safe_mean(energy_proxy_values),
        "energy_proxy_std": _safe_std(energy_proxy_values),
        "energy_proxy_jerk_mean": _safe_mean(energy_proxy_jerk_values),
        "energy_proxy_jerk_std": _safe_std(energy_proxy_jerk_values),
        "energy_proxy_rate_mean": _safe_mean(energy_proxy_rate_values),
        "energy_proxy_rate_std": _safe_std(energy_proxy_rate_values),
        "energy_proxy_jerk_rate_mean": _safe_mean(energy_proxy_jerk_rate_values),
        "energy_proxy_jerk_rate_std": _safe_std(energy_proxy_jerk_rate_values),
        "success_rate": (
            float(hinge_opportunity_success_count / hinge_opportunity_total_count)
            if hinge_opportunity_total_count > 0
            else float("nan")
        ),
        "success_event_count": hinge_opportunity_success_count,
        "success_opportunity_count": hinge_opportunity_total_count,
        "terminal_success_rate": (
            float(sum(terminal_success_flags) / total_episodes)
            if total_episodes > 0
            else float("nan")
        ),
        "collision_rate": (
            float(sum(collision_flags) / total_episodes)
            if total_episodes > 0
            else float("nan")
        ),
        "collision_rate_agents": (
            float(sum(collision_agents_flags) / total_episodes)
            if total_episodes > 0
            else float("nan")
        ),
        "collision_rate_lanelets": (
            float(sum(collision_lanelets_flags) / total_episodes)
            if total_episodes > 0
            else float("nan")
        ),
        "collision_rate_exit_segments": (
            float(sum(collision_exit_flags) / total_episodes)
            if total_episodes > 0
            else float("nan")
        ),
    }

    # Temporarily hide per-agent metrics and only keep aggregated follower averages.
    # for agent_id in resolved_followers:
    #     results[f"agent_{agent_id}_hinge_time_sec_mean"] = _safe_mean(
    #         hinge_times_by_agent[agent_id]
    #     )
    #     results[f"agent_{agent_id}_hinge_time_sec_std"] = _safe_std(
    #         hinge_times_by_agent[agent_id]
    #     )
    #     results[f"agent_{agent_id}_hinge_instant_speed_diff_mean"] = _safe_mean(
    #         hinge_speed_diffs_by_agent[agent_id]
    #     )
    #     results[f"agent_{agent_id}_hinge_instant_speed_diff_std"] = _safe_std(
    #         hinge_speed_diffs_by_agent[agent_id]
    #     )
    #     results[
    #         f"agent_{agent_id}_hinge_pre_dock_vel_angle_diff_deg_mean"
    #     ] = _safe_mean(hinge_pre_dock_vel_angle_diffs_by_agent[agent_id])
    #     results[
    #         f"agent_{agent_id}_hinge_pre_dock_vel_angle_diff_deg_std"
    #     ] = _safe_std(hinge_pre_dock_vel_angle_diffs_by_agent[agent_id])
    #     results[f"agent_{agent_id}_hinge_instant_angle_diff_deg_mean"] = _safe_mean(
    #         hinge_pre_dock_vel_angle_diffs_by_agent[agent_id]
    #     )
    #     results[f"agent_{agent_id}_hinge_instant_angle_diff_deg_std"] = _safe_std(
    #         hinge_pre_dock_vel_angle_diffs_by_agent[agent_id]
    #     )
    #     results[f"agent_{agent_id}_hinge_gate_angle_diff_deg_mean"] = _safe_mean(
    #         hinge_gate_angle_diffs_by_agent[agent_id]
    #     )
    #     results[f"agent_{agent_id}_hinge_gate_angle_diff_deg_std"] = _safe_std(
    #         hinge_gate_angle_diffs_by_agent[agent_id]
    #     )

    return results


def compute_validation_metrics(
    result_path: str,
    *,
    followers: Optional[List[int]] = None,
    dt: Optional[float] = None,
) -> Dict[str, object]:
    result_data = torch.load(result_path, map_location="cpu", weights_only=False)
    return compute_validation_metrics_from_object(
        result_data,
        followers=followers,
        dt=dt,
    )


def detect_input_format(loaded_object: Any) -> str:
    if isinstance(loaded_object, dict) and loaded_object.get("format") == "occt_traditional_validation":
        return "validation"
    if isinstance(loaded_object, dict) and "episodes" in loaded_object and "method" in loaded_object:
        return "validation"
    return "rollout"


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return value
    if isinstance(value, (int, str)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _parse_simple_rollout_yaml(config_path: Path) -> Dict[str, str]:
    config: Dict[str, str] = {}
    current_key: Optional[str] = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
            current_key = _normalize_method_name(stripped[:-1])
            continue
        if current_key is None:
            continue
        if stripped.startswith("rollout_path:"):
            rollout_path = stripped.split(":", 1)[1].strip().strip("'\"")
            config[current_key] = rollout_path
            current_key = None
    return config


def _load_method_rollout_config(data_dir: Path) -> Dict[str, str]:
    config_path = data_dir / "rollout_path.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"rollout_path.yaml not found in {data_dir}")

    if yaml is not None:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Invalid rollout_path.yaml format in {config_path}")
        config = {}
        for method_name, method_info in loaded.items():
            if not isinstance(method_info, dict) or "rollout_path" not in method_info:
                raise ValueError(
                    f"Method '{method_name}' in {config_path} must contain rollout_path."
                )
            config[_normalize_method_name(str(method_name))] = str(
                method_info["rollout_path"]
            ).strip()
        return config

    config = _parse_simple_rollout_yaml(config_path)
    if not config:
        raise ValueError(
            f"Unable to parse rollout_path.yaml without PyYAML: {config_path}"
        )
    return config


def _resolve_method_source_path(data_dir: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    return candidate if candidate.is_absolute() else (data_dir / candidate)


def _resolve_method_output_root(source_path: Path) -> Path:
    if source_path.is_dir():
        return source_path
    if source_path.parent.name == "rollouts":
        return source_path.parent.parent
    return source_path.parent


def _load_single_method_datasets(
    *,
    method: str,
    source_path: Path,
    followers: Optional[List[int]],
    dt: Optional[float],
) -> List[Dict[str, Any]]:
    if method in {"pid", "mppi"}:
        input_paths = resolve_input_paths([str(source_path)])
        datasets_by_method = _collect_method_datasets(
            input_paths,
            forced_format="validation",
            followers=followers,
            dt=dt,
            labels=[method] * len(input_paths),
        )
    else:
        if not source_path.exists():
            raise FileNotFoundError(f"Rollout file not found: {source_path}")
        datasets_by_method = _collect_method_datasets(
            [source_path],
            forced_format="rollout",
            followers=followers,
            dt=dt,
            labels=[method],
        )

    normalized_method = _normalize_method_name(method)
    if normalized_method not in datasets_by_method:
        raise ValueError(
            f"Loaded method '{normalized_method}' not found. "
            f"Available methods: {sorted(datasets_by_method.keys())}"
        )
    return datasets_by_method[normalized_method]


def _default_report_dir() -> Path:
    return Path.cwd() / "occt_metrics_reports"


def _collect_method_datasets(
    input_paths: Sequence[Path],
    *,
    forced_format: str,
    followers: Optional[List[int]],
    dt: Optional[float],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    datasets_by_method: Dict[str, List[Dict[str, Any]]] = {}
    if labels is not None and len(labels) != len(input_paths):
        raise ValueError(
            f"--labels count ({len(labels)}) must match resolved input file count ({len(input_paths)})."
        )

    for input_index, input_path in enumerate(input_paths):
        label = labels[input_index] if labels is not None else None
        loaded_object = torch.load(input_path, map_location="cpu", weights_only=False)
        source_format = forced_format
        if source_format == "auto":
            source_format = detect_input_format(loaded_object)

        if source_format == "validation":
            dataset = loaded_object
            if label is not None:
                dataset = {**dataset, "method": _normalize_method_name(label)}
        else:
            inferred_method = _infer_rollout_method_name(input_path, label)
            dataset = build_result_data_from_rollout(
                loaded_object,
                followers=followers or list(DEFAULT_FOLLOWERS),
                dt=DEFAULT_DT if dt is None else dt,
                method=inferred_method,
            )

        method = _normalize_method_name(str(dataset.get("method", "unknown")))
        datasets_by_method.setdefault(method, []).append(dataset)
    return datasets_by_method


def _build_method_reports(
    datasets: List[Dict[str, Any]],
    *,
    method: str,
    followers: Optional[List[int]],
    dt: Optional[float],
) -> Dict[str, List[Dict[str, Any]]]:
    method_episodes, episodes_by_road, resolved_dt, resolved_followers = (
        _prepare_method_episode_groups(
            datasets,
            followers=followers,
            dt=dt,
        )
    )

    per_road_rows = []
    for road_id in sorted(episodes_by_road):
        road_episodes = episodes_by_road[road_id]
        road_name = _extract_episode_road_name(road_episodes[0], road_id)
        bundle = _build_result_bundle(
            method=method,
            episodes=road_episodes,
            dt=resolved_dt,
            followers=resolved_followers,
            source_format="aggregated",
            road_name=road_name,
            requested_road_id=road_id,
            report_type="per_road",
            road_type=_get_road_type(road_id),
            road_ids=[road_id],
        )
        metrics = compute_validation_metrics_from_object(
            bundle,
            followers=resolved_followers,
            dt=resolved_dt,
        )
        metrics["report_type"] = "per_road"
        metrics["road_type"] = _get_road_type(road_id)
        per_road_rows.append(metrics)

    episodes_by_type: Dict[str, List[Dict[str, Any]]] = {}
    road_ids_by_type: Dict[str, List[int]] = {}
    for road_id, road_episodes in episodes_by_road.items():
        road_type = _get_road_type(road_id)
        episodes_by_type.setdefault(road_type, []).extend(road_episodes)
        road_ids_by_type.setdefault(road_type, []).append(road_id)

    per_type_rows = []
    for road_type in sorted(episodes_by_type):
        type_episodes = episodes_by_type[road_type]
        type_road_ids = sorted(set(road_ids_by_type[road_type]))
        bundle = _build_result_bundle(
            method=method,
            episodes=type_episodes,
            dt=resolved_dt,
            followers=resolved_followers,
            source_format="aggregated",
            road_name=road_type,
            requested_road_id=None,
            report_type="road_type",
            road_type=road_type,
            road_ids=type_road_ids,
        )
        metrics = compute_validation_metrics_from_object(
            bundle,
            followers=resolved_followers,
            dt=resolved_dt,
        )
        metrics["report_type"] = "road_type"
        metrics["road_type"] = road_type
        per_type_rows.append(metrics)

    overall_bundle = _build_result_bundle(
        method=method,
        episodes=method_episodes,
        dt=resolved_dt,
        followers=resolved_followers,
        source_format="aggregated",
        road_name="overall",
        requested_road_id=None,
        report_type="overall",
        road_type="all",
        road_ids=sorted(episodes_by_road.keys()),
    )
    overall_metrics = compute_validation_metrics_from_object(
        overall_bundle,
        followers=resolved_followers,
        dt=resolved_dt,
    )
    overall_metrics["report_type"] = "overall"
    overall_metrics["road_type"] = "all"
    overall_rows = [overall_metrics]

    return {
        "per_road": per_road_rows,
        "road_type": per_type_rows,
        "overall": overall_rows,
    }


def _prepare_method_episode_groups(
    datasets: List[Dict[str, Any]],
    *,
    followers: Optional[List[int]],
    dt: Optional[float],
) -> tuple[list[Dict[str, Any]], dict[int, list[Dict[str, Any]]], float, list[int]]:
    method_episodes = []
    resolved_dt = None
    resolved_followers = None

    for dataset in datasets:
        dataset_followers = _resolve_followers(dataset, followers)
        dataset_dt = float(dataset.get("dt", DEFAULT_DT) if dt is None else dt)
        if resolved_dt is None:
            resolved_dt = dataset_dt
        if resolved_followers is None:
            resolved_followers = dataset_followers

        fallback_road_id = dataset.get("requested_road_id")
        fallback_road_name = dataset.get("road_name")
        for episode in dataset.get("episodes", []):
            road_id = _extract_episode_road_id(episode, fallback_road_id)
            road_name = _extract_episode_road_name(episode, road_id, fallback_road_name)
            method_episodes.append(
                {
                    **episode,
                    "road_id": road_id,
                    "road_name": road_name,
                }
            )

    if resolved_dt is None:
        resolved_dt = DEFAULT_DT if dt is None else dt
    if resolved_followers is None:
        resolved_followers = list(DEFAULT_FOLLOWERS) if followers is None else followers

    episodes_by_road: Dict[int, List[Dict[str, Any]]] = {}
    for episode in method_episodes:
        road_id = _extract_episode_road_id(episode)
        if road_id is None:
            continue
        episodes_by_road.setdefault(road_id, []).append(episode)
    return method_episodes, episodes_by_road, float(resolved_dt), list(resolved_followers)


def _write_method_csv(report_dir: Path, method: str, report_groups: Dict[str, List[Dict[str, Any]]]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{method}_reports.csv"
    rows: List[Dict[str, Any]] = []
    for report_type in ("per_road", "road_type", "overall"):
        rows.extend(report_groups.get(report_type, []))
    if not rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["method", "report_type"])
        return csv_path

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(row.get(key)) for key in fieldnames})
    return csv_path


def _find_method_report_csv(data_dir: Path, method: str) -> Path:
    normalized_method = _normalize_method_name(method)
    matches = sorted(data_dir.glob(f"**/table/{normalized_method}_reports.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No CSV report found for method '{normalized_method}' under {data_dir}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple CSV reports found for method '{normalized_method}': "
            f"{[str(path) for path in matches]}"
        )
    return matches[0]


def _load_method_report_rows(report_csv_path: Path) -> List[Dict[str, str]]:
    with report_csv_path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def _select_comparison_row(
    rows: List[Dict[str, str]],
    *,
    report_type: str,
    road_type: Optional[str] = None,
) -> Dict[str, str]:
    for row in rows:
        if row.get("report_type") != report_type:
            continue
        if road_type is not None and row.get("road_type") != road_type:
            continue
        return row
    raise ValueError(
        f"No row found for report_type='{report_type}', road_type='{road_type}'."
    )


def _write_comparison_csv(output_path: Path, rows: List[Dict[str, str]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_csv_value(row.get(key)) for key in fieldnames})
    return output_path


def generate_comparison_csvs(
    *,
    data_dir: Path,
    methods: Sequence[str],
) -> List[Path]:
    normalized_methods = _ordered_methods([_normalize_method_name(method) for method in methods])
    report_rows_by_method: Dict[str, List[Dict[str, str]]] = {}
    for method in normalized_methods:
        report_csv_path = _find_method_report_csv(data_dir, method)
        report_rows_by_method[method] = _load_method_report_rows(report_csv_path)

    comparison_specs = [
        ("comparison_roundabout.csv", "road_type", "roundabout"),
        ("comparison_right_angle_turn.csv", "road_type", "right_angle_turn"),
        ("comparison_s_curve.csv", "road_type", "s_curve"),
        ("comparison_overall.csv", "overall", None),
    ]

    saved_paths: List[Path] = []
    for file_name, report_type, road_type in comparison_specs:
        rows: List[Dict[str, str]] = []
        for method in normalized_methods:
            selected_row = dict(
                _select_comparison_row(
                    report_rows_by_method[method],
                    report_type=report_type,
                    road_type=road_type,
                )
            )
            selected_row["display_name"] = _get_method_display_name(method)
            rows.append(selected_row)
        saved_paths.append(_write_comparison_csv(data_dir / file_name, rows))
    return saved_paths


def _print_report_group(method: str, report_type: str, rows: List[Dict[str, Any]]) -> None:
    display_name = _get_method_display_name(method)
    print(f"\n##### {display_name} {report_type} #####")
    for row in rows:
        if report_type == "per_road":
            label = f"{display_name} road_{row.get('requested_road_id')} {row.get('road_name')}"
        elif report_type == "road_type":
            label = f"{display_name} road_type_{row.get('road_type')}"
        else:
            label = f"{display_name} overall"
        print_metrics(label, row)


def print_metrics(label: str, metrics: Dict[str, object]) -> None:
    print(f"\n=== {label} ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def _select_representative_episode(
    episodes: List[Dict[str, Any]],
    follower_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    successful_episodes = [
        episode
        for episode in episodes
        if _compute_episode_terminal_status(
            episode, follower_ids=follower_ids
        )["final_success"]
    ]
    if successful_episodes:
        return successful_episodes[0]
    return episodes[0]


def _get_agent_color(agent_id: int) -> str:
    return AGENT_COLORS.get(agent_id, "#4C72B0")


def _get_plot_method_line_palette(metric_key: str, count: int) -> List[str]:
    palette = PLOT_METHOD_LINE_PALETTES.get(metric_key)
    if palette is None:
        return [_get_agent_color(agent_id) for agent_id in range(count)]
    if len(palette) >= count:
        return palette[:count]
    return [palette[idx % len(palette)] for idx in range(count)]


def _default_plot_dir() -> Path:
    return DEFAULT_PLOT_DIR


def _style_cn_axes(
    ax,
    *,
    title: str = None,
    y_label: str = None,
    x_label: str = "时间 (s)",
    show_legend: bool = True,
) -> None:
    if title:
        ax.set_title(title, fontproperties=font_prop_chinese, fontsize=font_size_title)
    if y_label:
        ax.set_ylabel(y_label, fontproperties=font_prop_chinese, fontsize=font_size_label)
    if x_label:
        ax.set_xlabel(x_label, fontproperties=font_prop_chinese, fontsize=font_size_label)
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


def _save_pdf_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.05,
    )
    plt.close(fig)


def _draw_fill_band(
    ax,
    *,
    time_axis: np.ndarray,
    center_values: np.ndarray,
    upper_values: np.ndarray,
    lower_values: np.ndarray,
    color: str,
    mean_label: str,
    band_label: str,
) -> None:
    ax.plot(
        time_axis,
        center_values,
        label=mean_label,
        linewidth=1.2,
        color=color,
        alpha=0.95,
    )
    ax.plot(time_axis, upper_values, linewidth=0.6, color=color, alpha=0.35)
    ax.plot(time_axis, lower_values, linewidth=0.6, color=color, alpha=0.35)
    ax.fill_between(
        time_axis,
        lower_values,
        upper_values,
        color=color,
        alpha=0.14,
        label=band_label,
    )


def _masked_axis_reduce(
    value_matrix: np.ndarray,
    reducer,
) -> np.ndarray:
    masked_values = np.ma.masked_invalid(value_matrix)
    reduced = reducer(masked_values, axis=1)
    if isinstance(reduced, np.ma.MaskedArray):
        return reduced.filled(np.nan)
    return np.asarray(reduced, dtype=float)


def _mask_metric_matrix_by_phase(
    value_matrix: np.ndarray,
    phase_mask: torch.Tensor,
) -> np.ndarray:
    masked_values = np.array(value_matrix, copy=True, dtype=float)
    invalid_mask = ~torch.as_tensor(phase_mask, dtype=torch.bool).detach().cpu().numpy()
    masked_values[invalid_mask] = np.nan
    return masked_values


def _infer_num_agents(episode: Dict[str, Any]) -> int:
    info = episode["info"]
    for key in ("vel", "pos", "act_acc", "act_steer", "error_space", "hinge_dis"):
        if key not in info:
            continue
        value = _squeeze_trailing_unit_dim(info[key])
        if value.ndim >= 2:
            return int(value.shape[1])
    raise ValueError("无法从episode中推断agent数量。")


def _filter_valid_agent_ids(
    candidate_agent_ids: Sequence[int], num_agents: int
) -> List[int]:
    valid_agent_ids = [
        int(agent_id)
        for agent_id in candidate_agent_ids
        if 0 <= int(agent_id) < num_agents
    ]
    if not valid_agent_ids:
        raise ValueError("给定绘图agent索引超出当前episode的agent范围。")
    return valid_agent_ids


def _format_agent_suffix(agent_ids: Sequence[int]) -> str:
    normalized = [int(agent_id) for agent_id in agent_ids]
    if len(normalized) == 1:
        return f"agent{normalized[0]}"
    if normalized == list(range(normalized[0], normalized[-1] + 1)):
        return f"agents{normalized[0]}-{normalized[-1]}"
    return "agents" + "_".join(str(agent_id) for agent_id in normalized)


def _flatten_scalar_series(value: torch.Tensor) -> torch.Tensor:
    tensor = value.float()
    if tensor.ndim == 1:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)[:, 0]


def _extract_speed_series(episode: Dict[str, Any], agent_id: int) -> torch.Tensor:
    agent_vel = _extract_agent_series(episode, "vel", agent_id).float()
    if agent_vel.ndim == 1:
        return agent_vel.abs()
    agent_vel = agent_vel.reshape(agent_vel.shape[0], -1)
    return torch.linalg.norm(agent_vel, dim=-1)


def _extract_acc_series(
    episode: Dict[str, Any], agent_id: int, *, dt: float
) -> torch.Tensor:
    info = episode["info"]
    if "act_acc" in info:
        return _flatten_scalar_series(_extract_agent_series(episode, "act_acc", agent_id))
    speed_series = _extract_speed_series(episode, agent_id)
    acceleration = torch.diff(speed_series, prepend=speed_series[:1]) / dt
    return acceleration.float()


def _extract_jerk_series(
    episode: Dict[str, Any], agent_id: int, *, dt: float
) -> torch.Tensor:
    info = episode["info"]
    if "command_jerk" in info:
        return _flatten_scalar_series(
            _extract_agent_series(episode, "command_jerk", agent_id)
        ).float()
    acc_series = _extract_acc_series(episode, agent_id, dt=dt)
    jerk = torch.diff(acc_series, prepend=acc_series[:1]) / dt
    return jerk.float()


def _extract_steer_series(episode: Dict[str, Any], agent_id: int) -> torch.Tensor:
    steer_series = _flatten_scalar_series(_extract_agent_series(episode, "act_steer", agent_id))
    return steer_series * (180.0 / np.pi)


def _extract_heading_series(episode: Dict[str, Any], agent_id: int) -> torch.Tensor:
    heading_series = _flatten_scalar_series(_extract_agent_series(episode, "rot", agent_id))
    heading_np = heading_series.detach().cpu().numpy()
    heading_np = np.unwrap(heading_np)
    heading_deg = np.rad2deg(heading_np)
    return torch.as_tensor(heading_deg, dtype=torch.float32)


def _extract_steering_rate_series(
    episode: Dict[str, Any], agent_id: int, *, dt: float
) -> torch.Tensor:
    info = episode["info"]
    if "steering_rate_deg" in info:
        return _flatten_scalar_series(
            _extract_agent_series(episode, "steering_rate_deg", agent_id)
        ).float()
    steer_series_rad = _flatten_scalar_series(
        _extract_agent_series(episode, "act_steer", agent_id)
    ).float()
    steering_rate = torch.diff(steer_series_rad, prepend=steer_series_rad[:1]) / dt
    return steering_rate * (180.0 / np.pi)


def _plot_single_metric_pdf(
    *,
    time_axis: np.ndarray,
    series_specs: List[Dict[str, Any]],
    title: str,
    y_label: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    for spec in series_specs:
        ax.plot(
            time_axis,
            spec["values"],
            label=spec["label"],
            linewidth=1.2,
            color=spec["color"],
            alpha=0.95,
        )
    _style_cn_axes(ax, title=title, y_label=y_label, show_legend=True)
    _save_pdf_figure(fig, output_path)
    return output_path


def _plot_mean_band_metric_pdf(
    *,
    time_axis: np.ndarray,
    value_matrix: np.ndarray,
    title: str,
    y_label: str,
    output_path: Path,
    color: str,
    mean_label: str,
    band_label: str,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    _draw_fill_band(
        ax,
        time_axis=time_axis,
        center_values=np.mean(value_matrix, axis=1),
        upper_values=np.max(value_matrix, axis=1),
        lower_values=np.min(value_matrix, axis=1),
        color=color,
        mean_label=mean_label,
        band_label=band_label,
    )
    _style_cn_axes(ax, title=title, y_label=y_label, show_legend=True)
    _save_pdf_figure(fig, output_path)
    return output_path


def _plot_plot_method_metric_pdf(
    *,
    time_axis: np.ndarray,
    value_matrix: np.ndarray,
    agent_ids: Sequence[int],
    palette_key: str,
    title: Optional[str] = None,
    y_label: str,
    output_path: Path,
    fill_style: bool,
    fill_color: str,
    y_limits: Optional[tuple[float, float]] = None,
    background_mask: Optional[np.ndarray] = None,
    background_dt: Optional[float] = None,
    background_label: str = "阶段",
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)

    background_handle = None
    if background_mask is not None and background_dt is not None:
        background_handle = _shade_boolean_background(
            ax,
            torch.as_tensor(background_mask, dtype=torch.bool),
            background_dt,
            label=background_label,
        )

    if fill_style:
        _draw_fill_band(
            ax,
            time_axis=time_axis,
            center_values=_masked_axis_reduce(value_matrix, np.ma.mean),
            upper_values=_masked_axis_reduce(value_matrix, np.ma.max),
            lower_values=_masked_axis_reduce(value_matrix, np.ma.min),
            color=fill_color,
            mean_label="车辆1-3均值",
            band_label="车辆1-3波动带",
        )
    else:
        line_colors = _get_plot_method_line_palette(palette_key, len(agent_ids))
        for idx, agent_id in enumerate(agent_ids):
            ax.plot(
                time_axis,
                value_matrix[:, idx],
                label=f"车辆{agent_id}",
                linewidth=1.2,
                color=line_colors[idx],
                alpha=0.95,
            )

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if background_handle is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles.append(background_handle)
        labels.append(background_label)
        ax.legend(
            handles,
            labels,
            loc="best",
            fontsize=font_size_legend,
            prop=font_prop_chinese,
            handlelength=1.8,
            borderpad=0.2,
            labelspacing=0.2,
        )
        _style_cn_axes(ax, title=title, y_label=y_label, show_legend=False)
    else:
        _style_cn_axes(ax, title=title, y_label=y_label, show_legend=True)

    _save_pdf_figure(fig, output_path)
    return output_path


def _build_series_specs_from_agents(
    *,
    episode: Dict[str, Any],
    agent_ids: Sequence[int],
    extractor,
    label_prefix: str = "车辆",
    dt: Optional[float] = None,
) -> List[Dict[str, Any]]:
    series_specs = []
    for agent_id in agent_ids:
        if dt is None:
            values = extractor(episode, agent_id)
        else:
            values = extractor(episode, agent_id, dt=dt)
        series_specs.append(
            {
                "label": f"{label_prefix}{agent_id}",
                "values": values.detach().cpu().numpy(),
                "color": _get_agent_color(agent_id),
            }
        )
    return series_specs


def plot_representative_road_curves(
    datasets: List[Dict[str, Any]],
    *,
    method: str,
    followers: Optional[List[int]],
    dt: Optional[float],
    plot_dir: Path,
    road_ids: Sequence[int] = (0,),
    plot_method_fill_stype: bool = True,
) -> List[Path]:
    _, episodes_by_road, resolved_dt, resolved_followers = _prepare_method_episode_groups(
        datasets,
        followers=followers,
        dt=dt,
    )
    target_road_ids = [int(road_id) for road_id in road_ids]
    missing_road_ids = [road_id for road_id in target_road_ids if road_id not in episodes_by_road]
    if missing_road_ids:
        raise ValueError(
            f"Method '{method}' is missing representative roads {missing_road_ids}."
        )

    representative_episodes = {
        road_id: _select_representative_episode(
            episodes_by_road[road_id],
            follower_ids=resolved_followers,
        )
        for road_id in target_road_ids
    }

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for road_id in target_road_ids:
        episode = representative_episodes[road_id]
        method_label = _get_method_display_name(method)
        num_agents = _infer_num_agents(episode)
        error_agent_ids = _filter_valid_agent_ids(resolved_followers, num_agents)
        hinge_agent_ids = _filter_valid_agent_ids(resolved_followers, num_agents)
        control_agent_ids = _filter_valid_agent_ids([1, 2, 3], num_agents)
        time_axis = (
            torch.arange(int(episode["num_steps"]), dtype=torch.float32) * resolved_dt
        ).numpy()
        style_suffix = "fill" if plot_method_fill_stype else "lines"
        hinge_status_matrix = _squeeze_trailing_unit_dim(
            episode["info"]["hinge_status"][:, error_agent_ids]
        ).bool()
        platoon_phase_mask = ~hinge_status_matrix
        hinge_phase_mask = hinge_status_matrix.any(dim=1)

        longitudinal_values = (
            episode["info"]["error_space"][:, error_agent_ids, 0]
            .float()
            .detach()
            .cpu()
            .numpy()
        )
        longitudinal_values = _mask_metric_matrix_by_phase(
            longitudinal_values,
            platoon_phase_mask,
        )
        longitudinal_path = plot_dir / (
            f"longitudinal_error_{method}_road{road_id}_"
            f"{_format_agent_suffix(error_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=longitudinal_values,
                agent_ids=error_agent_ids,
                palette_key="longitudinal_error",
                #title=f"{method_label} 道路{road_id} 编队纵向误差",
                y_label="纵向误差 (m)",
                output_path=longitudinal_path,
                fill_style=plot_method_fill_stype,
                fill_color="#9467bd",
                background_mask=hinge_phase_mask.detach().cpu().numpy(),
                background_dt=resolved_dt,
                background_label="铰接阶段",
            )
        )

        lateral_values = _squeeze_trailing_unit_dim(
            episode["info"]["distance_ref"][:, error_agent_ids]
        ).float().detach().cpu().numpy()
        lateral_values = _mask_metric_matrix_by_phase(
            lateral_values,
            platoon_phase_mask,
        )
        lateral_path = plot_dir / (
            f"lateral_error_{method}_road{road_id}_"
            f"{_format_agent_suffix(error_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=lateral_values,
                agent_ids=error_agent_ids,
                palette_key="lateral_error",
                #title=f"{method_label} 道路{road_id} 编队横向误差",
                y_label="横向误差 (m)",
                output_path=lateral_path,
                fill_style=plot_method_fill_stype,
                fill_color="#8c564b",
                background_mask=hinge_phase_mask.detach().cpu().numpy(),
                background_dt=resolved_dt,
                background_label="铰接阶段",
            )
        )

        hinge_phase_mask = _extract_agent_series(
            episode, "hinge_status", hinge_agent_ids[0]
        ).bool()
        for agent_id in hinge_agent_ids[1:]:
            hinge_phase_mask |= _extract_agent_series(
                episode, "hinge_status", agent_id
            ).bool()
        hinge_distance_matrix = np.stack(
            [
                _extract_hinge_distance_series(episode, agent_id).detach().cpu().numpy()
                for agent_id in hinge_agent_ids
            ],
            axis=1,
        )
        hinge_path = plot_dir / (
            f"hinge_dis_{method}_road{road_id}_"
            f"{_format_agent_suffix(hinge_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=hinge_distance_matrix,
                agent_ids=hinge_agent_ids,
                palette_key="hinge_dis",
                #title=f"{method_label} 道路{road_id} 铰接距离",
                y_label="铰接距离 (m)",
                output_path=hinge_path,
                fill_style=plot_method_fill_stype,
                fill_color="#17becf",
                y_limits=(0.0, float(np.max(hinge_distance_matrix)) + 0.2),
                background_mask=hinge_phase_mask.detach().cpu().numpy(),
                background_dt=resolved_dt,
                background_label="铰接可用",
            )
        )

        hinge_status_matrix = np.stack(
            [
                _extract_agent_series(episode, "hinge_status", agent_id)
                .float()
                .detach()
                .cpu()
                .numpy()
                for agent_id in hinge_agent_ids
            ],
            axis=1,
        )
        hinge_status_path = plot_dir / (
            f"hinge_status_{method}_road{road_id}_"
            f"{_format_agent_suffix(hinge_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=hinge_status_matrix,
                agent_ids=hinge_agent_ids,
                palette_key="hinge_status",
                #title=f"{method_label} 道路{road_id} 铰接状态",
                y_label="铰接状态",
                output_path=hinge_status_path,
                fill_style=plot_method_fill_stype,
                fill_color="#bcbd22",
                y_limits=(-0.05, 1.05),
            )
        )

        speed_matrix = np.stack(
            [
                _extract_speed_series(episode, agent_id).detach().cpu().numpy()
                for agent_id in control_agent_ids
            ],
            axis=1,
        )
        speed_path = plot_dir / (
            f"speed_{method}_road{road_id}_"
            f"{_format_agent_suffix(control_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=speed_matrix,
                agent_ids=control_agent_ids,
                palette_key="speed",
                #title=f"{method_label} 道路{road_id} 车辆速度",
                y_label="速度 (m/s)",
                output_path=speed_path,
                fill_style=plot_method_fill_stype,
                fill_color="#17becf",
            )
        )

        acceleration_matrix = np.stack(
            [
                _extract_acc_series(episode, agent_id, dt=resolved_dt)
                .detach()
                .cpu()
                .numpy()
                for agent_id in control_agent_ids
            ],
            axis=1,
        )
        acceleration_path = plot_dir / (
            f"acceleration_{method}_road{road_id}_"
            f"{_format_agent_suffix(control_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=acceleration_matrix,
                agent_ids=control_agent_ids,
                palette_key="acceleration",
                #title=f"{method_label} 道路{road_id} 车辆加速度",
                y_label="加速度 (m/s^2)",
                output_path=acceleration_path,
                fill_style=plot_method_fill_stype,
                fill_color="#e377c2",
            )
        )

        steering_matrix = np.stack(
            [
                _extract_steer_series(episode, agent_id).detach().cpu().numpy()
                for agent_id in control_agent_ids
            ],
            axis=1,
        )
        steering_path = plot_dir / (
            f"steering_angle_{method}_road{road_id}_"
            f"{_format_agent_suffix(control_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=steering_matrix,
                agent_ids=control_agent_ids,
                palette_key="steering_angle",
                #title=f"{method_label} 道路{road_id} 前轮转角",
                y_label="前轮转角 (deg)",
                output_path=steering_path,
                fill_style=plot_method_fill_stype,
                fill_color="#7f7f7f",
            )
        )

        heading_matrix = np.stack(
            [
                _extract_heading_series(episode, agent_id).detach().cpu().numpy()
                for agent_id in control_agent_ids
            ],
            axis=1,
        )
        heading_path = plot_dir / (
            f"heading_angle_{method}_road{road_id}_"
            f"{_format_agent_suffix(control_agent_ids)}_{style_suffix}.pdf"
        )
        saved_paths.append(
            _plot_plot_method_metric_pdf(
                time_axis=time_axis,
                value_matrix=heading_matrix,
                agent_ids=control_agent_ids,
                palette_key="heading_angle",
                #title=f"{method_label} 道路{road_id} 车辆航向角",
                y_label="航向角 (deg)",
                output_path=heading_path,
                fill_style=plot_method_fill_stype,
                fill_color="#1f77b4",
            )
        )

    return saved_paths


def _collect_method_distribution_stats(
    datasets: List[Dict[str, Any]],
    *,
    followers: Optional[List[int]],
    dt: Optional[float],
) -> Dict[str, List[float]]:
    method_episodes, _, resolved_dt, resolved_followers = _prepare_method_episode_groups(
        datasets,
        followers=followers,
        dt=dt,
    )

    platoon_ttc_episode_mins: List[float] = []
    hinge_times: List[float] = []
    hinge_speed_diffs: List[float] = []
    hinge_gate_angle_diffs: List[float] = []
    control_acc_values: List[float] = []
    control_jerk_values: List[float] = []
    control_steering_rate_values: List[float] = []

    for episode in method_episodes:
        info = episode["info"]
        hinge_status = _squeeze_trailing_unit_dim(
            info["hinge_status"][:, resolved_followers]
        ).bool()
        agent_hinge_status = _squeeze_trailing_unit_dim(
            info["agent_hinge_status"][:, resolved_followers]
        ).bool()
        s_all = _squeeze_trailing_unit_dim(info["s"]).float()
        vel_all = info["vel"].float()
        speed_all = torch.linalg.norm(vel_all, dim=-1)
        vel = info["vel"][:, resolved_followers, :].float()
        hinge_vel = (
            info["hinge_vel"][:, resolved_followers, :].float()
            if "hinge_vel" in info
            else None
        )
        error_vel = (
            info["error_vel"][:, resolved_followers, :].float()
            if "error_vel" in info
            else None
        )
        hinge_gate_angle = (
            _squeeze_trailing_unit_dim(
                info["hinge_gate_angle_diff_deg"][:, resolved_followers]
            ).float()
            if "hinge_gate_angle_diff_deg" in info
            else None
        )

        episode_acc_values = []
        episode_jerk_values = []
        episode_steering_rate_values = []
        for follower_idx, agent_id in enumerate(resolved_followers):
            active_control_mask = ~agent_hinge_status[:, follower_idx]
            if not active_control_mask.any():
                continue
            if "act_acc" in info:
                acc_series = _squeeze_trailing_unit_dim(
                    info["act_acc"][:, agent_id]
                ).float().reshape(-1)
            else:
                acc_series = _compute_signal_rate(speed_all[:, agent_id], resolved_dt)
            jerk_series = _compute_signal_rate(acc_series, resolved_dt)
            if "steering_rate_abs_deg" in info:
                steering_rate_abs_series = _squeeze_trailing_unit_dim(
                    info["steering_rate_abs_deg"][:, agent_id]
                ).float().reshape(-1)
            else:
                steer_series = _squeeze_trailing_unit_dim(
                    info["act_steer"][:, agent_id]
                ).float().reshape(-1)
                steering_rate_abs_series = (
                    _compute_signal_rate(steer_series, resolved_dt).abs()
                    * (180.0 / torch.pi)
                )
            episode_acc_values.append(
                float(acc_series[active_control_mask].abs().mean().item())
            )
            episode_jerk_values.append(
                float(jerk_series[active_control_mask].abs().mean().item())
            )
            episode_steering_rate_values.append(
                float(steering_rate_abs_series[active_control_mask].mean().item())
            )
        if episode_acc_values:
            control_acc_values.append(float(np.mean(episode_acc_values)))
        if episode_jerk_values:
            control_jerk_values.append(float(np.mean(episode_jerk_values)))
        if episode_steering_rate_values:
            control_steering_rate_values.append(float(np.mean(episode_steering_rate_values)))

        platoon_mask = ~hinge_status
        episode_ttc_candidates = []

        for follower_idx, agent_id in enumerate(resolved_followers):
            platoon_step_mask = platoon_mask[:, follower_idx]
            if platoon_step_mask.any():
                ego_speed = speed_all[:, agent_id]
                ego_s = s_all[:, agent_id]

                front_agent_id = agent_id - 1
                if front_agent_id >= 0:
                    front_speed = speed_all[:, front_agent_id]
                    front_s = s_all[:, front_agent_id]
                    front_distance = front_s - ego_s
                    front_closing_speed = ego_speed - front_speed
                    front_ttc_mask = (
                        platoon_step_mask
                        & (front_distance > 0.0)
                        & (front_closing_speed > 0.0)
                    )
                    if front_ttc_mask.any():
                        episode_ttc_candidates.extend(
                            (
                                front_distance[front_ttc_mask]
                                / front_closing_speed[front_ttc_mask].clamp_min(1e-6)
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )

                back_agent_id = agent_id + 1
                if back_agent_id < s_all.shape[1]:
                    back_speed = speed_all[:, back_agent_id]
                    back_s = s_all[:, back_agent_id]
                    back_distance = ego_s - back_s
                    back_closing_speed = back_speed - ego_speed
                    back_ttc_mask = (
                        platoon_step_mask
                        & (back_distance > 0.0)
                        & (back_closing_speed > 0.0)
                    )
                    if back_ttc_mask.any():
                        episode_ttc_candidates.extend(
                            (
                                back_distance[back_ttc_mask]
                                / back_closing_speed[back_ttc_mask].clamp_min(1e-6)
                            )
                            .detach()
                            .cpu()
                            .tolist()
                        )

            for hinge_segment_start, _, success_index in _iter_hinge_opportunities(
                hinge_status[:, follower_idx],
                agent_hinge_status[:, follower_idx],
            ):
                if success_index is None:
                    continue

                metric_index = max(success_index - 1, 0)
                hinge_times.append(
                    float((success_index - hinge_segment_start + 1) * resolved_dt)
                )

                if hinge_vel is not None:
                    hinge_speed_diffs.append(
                        float(
                            torch.abs(
                                torch.linalg.norm(vel[metric_index, follower_idx])
                                - torch.linalg.norm(
                                    hinge_vel[metric_index, follower_idx]
                                )
                            ).item()
                        )
                    )
                elif error_vel is not None:
                    hinge_speed_diffs.append(
                        float(torch.abs(error_vel[metric_index, follower_idx, 0]).item())
                    )

                if hinge_gate_angle is not None:
                    hinge_gate_angle_diffs.append(
                        float(hinge_gate_angle[success_index, follower_idx].item())
                    )

        if episode_ttc_candidates:
            platoon_ttc_episode_mins.append(float(min(episode_ttc_candidates)))

    return {
        "ttc": platoon_ttc_episode_mins,
        "hinge_time": hinge_times,
        "hinge_speed_diff": hinge_speed_diffs,
        "hinge_gate_angle": hinge_gate_angle_diffs,
        "control_acc": control_acc_values,
        "control_jerk": control_jerk_values,
        "control_steering_rate": control_steering_rate_values,
    }


def _plot_boxplot_metric_pdf(
    *,
    method_value_map: Dict[str, List[float]],
    title: str,
    y_label: str,
    output_path: Path,
) -> Path:
    valid_items = [(method, values) for method, values in method_value_map.items() if values]
    if not valid_items:
        raise ValueError(f"{title} 没有可绘制的数据。")

    methods = [_get_method_display_name(method) for method, _ in valid_items]
    values = [series for _, series in valid_items]
    colors = [_get_method_color(method) for method, _ in valid_items]

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    boxplot = ax.boxplot(
        values,
        patch_artist=True,
        tick_labels=methods,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.0},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
        boxprops={"linewidth": 0.8},
    )
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    _style_cn_axes(ax, title=title, y_label=y_label, show_legend=False)
    _save_pdf_figure(fig, output_path)
    return output_path


def _plot_grouped_scaled_bar_metrics_pdf(
    *,
    metric_method_value_maps: List[Dict[str, object]],
    output_path: Path,
) -> Path:
    methods = _ordered_methods(
        list(
            dict.fromkeys(
                method
                for metric_spec in metric_method_value_maps
                for method in metric_spec["method_value_map"].keys()
            )
        )
    )
    group_centers = np.arange(len(metric_method_value_maps), dtype=float)
    bar_width = 0.18
    offsets = np.linspace(
        -bar_width * (len(methods) - 1) / 2.0,
        bar_width * (len(methods) - 1) / 2.0,
        len(methods),
    )

    fig_width = max(3.4, 0.95 * len(metric_method_value_maps) + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 2), constrained_layout=True)
    legend_handles = []
    legend_labels = []

    for method_index, method in enumerate(methods):
        color = _get_method_color(method)
        bar_positions = group_centers + offsets[method_index]
        scaled_heights = []
        true_values = []
        for metric_spec in metric_method_value_maps:
            method_value_map = metric_spec["method_value_map"]
            metric_values = method_value_map.get(method, [])
            mean_value = (
                float(np.mean(metric_values)) if metric_values else float("nan")
            )
            true_values.append(mean_value)

            group_means = [
                float(np.mean(values))
                for values in method_value_map.values()
                if values
            ]
            group_max = max(group_means) if group_means else float("nan")
            if np.isnan(mean_value) or np.isnan(group_max) or group_max <= 0.0:
                scaled_heights.append(0.0)
            else:
                scaled_heights.append(mean_value / group_max * 0.8)

        bars = ax.bar(
            bar_positions,
            scaled_heights,
            width=bar_width,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
            label=_get_method_display_name(method),
        )
        if method_index == 0:
            legend_handles.extend(bars)
            legend_labels.extend([method.upper()] * len(bars))
        for bar, true_value, scaled_height in zip(bars, true_values, scaled_heights):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                scaled_height + 0.025,
                "NA" if np.isnan(true_value) else f"{true_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=font_size_tick,
                fontproperties=font_prop_chinese,
                rotation=0,
            )

    ax.set_ylim(0.0, 1.0)
    #ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks([])
    #ax.set_ylabel("分数", fontproperties=font_prop_chinese, fontsize=font_size_label)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [metric_spec["group_label"] for metric_spec in metric_method_value_maps],
        fontproperties=font_prop_chinese,
        fontsize=font_size_tick,
    )
    ax.tick_params(
        axis="both",
        labelsize=font_size_tick,
        pad=1,
        direction="in",
        top=False,
        right=False,
        labelfontfamily="Times New Roman",
    )
    ax.margins(x=0.08)
    ax.legend(
        handles=[
                    plt.Rectangle((0, 0), 1, 1, facecolor=_get_method_color(method), alpha=0.78, edgecolor="white", linewidth=0.6)
                    for method in methods
                ],
        labels=[_get_method_display_name(method) for method in methods],
        loc="upper center",
        ncol=len(methods),
        fontsize=font_size_legend,
        prop=font_prop_chinese,
        frameon=False,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.4,
    )
    _save_pdf_figure(fig, output_path)
    return output_path


def plot_overall_metric_bars(
    datasets_by_method: Dict[str, List[Dict[str, Any]]],
    *,
    followers: Optional[List[int]],
    dt: Optional[float],
    plot_dir: Path,
    method_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    selected_methods = _ordered_methods(
        method_names if method_names is not None else list(datasets_by_method.keys())
    )
    stats_by_method = {
        method: _collect_method_distribution_stats(
            datasets_by_method[method],
            followers=followers,
            dt=dt,
        )
        for method in selected_methods
        if method in datasets_by_method
    }
    if not stats_by_method:
        raise ValueError("没有可用于多方法对比绘图的方法数据。")

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    saved_paths.append(
        _plot_boxplot_metric_pdf(
            method_value_map={method: stats["ttc"] for method, stats in stats_by_method.items()},
            title="最小TTC安全性对比",
            y_label="最小TTC (s)",
            output_path=plot_dir / "comparison_ttc_boxplot.pdf",
        )
    )
    saved_paths.append(
        _plot_grouped_scaled_bar_metrics_pdf(
            metric_method_value_maps=[
                {
                    "group_label": "铰接速度差 (m/s)",
                    "method_value_map": {
                        method: stats["hinge_speed_diff"]
                        for method, stats in stats_by_method.items()
                    },
                },
                {
                    "group_label": "铰接角度差 (deg)",
                    "method_value_map": {
                        method: stats["hinge_gate_angle"]
                        for method, stats in stats_by_method.items()
                    },
                },
                {
                    "group_label": "铰接时间 (s)",
                    "method_value_map": {
                        method: stats["hinge_time"]
                        for method, stats in stats_by_method.items()
                    },
                },
                {
                    "group_label": "纵向加速度",
                    "method_value_map": {
                        method: stats["control_acc"]
                        for method, stats in stats_by_method.items()
                    },
                },
                {
                    "group_label": "纵向Jerk",
                    "method_value_map": {
                        method: stats["control_jerk"]
                        for method, stats in stats_by_method.items()
                    },
                },
                {
                    "group_label": "转角变化率",
                    "method_value_map": {
                        method: stats["control_steering_rate"]
                        for method, stats in stats_by_method.items()
                    },
                },
            ],
            output_path=plot_dir / "comparison_hinge_metrics_bar.pdf",
        )
    )
    return saved_paths


def _ordered_methods(methods: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(methods))


def _get_method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#4C72B0")


def _extract_agent_series(
    episode: Dict[str, Any], key: str, agent_id: int
) -> torch.Tensor:
    info = episode["info"]
    value = info[key]
    value = _squeeze_trailing_unit_dim(value)
    if value.ndim == 1:
        return value.float()
    if value.ndim == 2:
        return value[:, agent_id].float()
    if value.ndim == 3:
        return value[:, agent_id].float()
    raise ValueError(f"Unsupported tensor shape for key '{key}': {tuple(value.shape)}")


def _extract_hinge_distance_series(
    episode: Dict[str, Any], agent_id: int
) -> torch.Tensor:
    info = episode["info"]
    if "hinge_pos" in info:
        hinge_pos = info["hinge_pos"][:, agent_id].float()
        pos = info["pos"][:, agent_id].float()
        return torch.linalg.norm(hinge_pos - pos, dim=-1)
    if "hinge_dis" in info:
        return _extract_agent_series(episode, "hinge_dis", agent_id).float()
    raise KeyError("Neither hinge_pos nor hinge_dis found in episode info.")


def _boolean_union_availability(
    representative_episodes: Dict[str, Dict[str, Any]],
    *,
    agent_id: int,
) -> tuple[torch.Tensor, float]:
    max_steps = 0
    common_dt = None
    for episode in representative_episodes.values():
        max_steps = max(max_steps, int(episode["num_steps"]))
        episode_dt = float(episode.get("dt", DEFAULT_DT))
        if common_dt is None:
            common_dt = episode_dt
    availability = torch.zeros(max_steps, dtype=torch.bool)
    for episode in representative_episodes.values():
        status = _extract_agent_series(episode, "hinge_status", agent_id).bool()
        availability[: status.shape[0]] |= status
    return availability, float(common_dt if common_dt is not None else DEFAULT_DT)


def _shade_boolean_background(ax, mask: torch.Tensor, dt: float, label: str) -> None:
    import matplotlib.patches as mpatches

    if mask.numel() == 0:
        return
    start_idx = None
    for idx, is_active in enumerate(mask.tolist()):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            ax.axvspan(start_idx * dt, idx * dt, color="#8fd694", alpha=0.18)
            start_idx = None
    if start_idx is not None:
        ax.axvspan(start_idx * dt, mask.shape[0] * dt, color="#8fd694", alpha=0.18)
    return mpatches.Patch(color="#8fd694", alpha=0.18, label=label)


def plot_method_transition_comparison(
    datasets_by_method: Dict[str, List[Dict[str, Any]]],
    *,
    method_names: Optional[Sequence[str]],
    road_id: int,
    agent_id: int,
    plot_dir: Path,
) -> List[Path]:
    import matplotlib.pyplot as plt

    selected_methods = _ordered_methods(
        method_names if method_names is not None else list(datasets_by_method.keys())
    )
    if len(selected_methods) < 2:
        raise ValueError("Method comparison plotting requires at least two methods.")

    representative_episodes: Dict[str, Dict[str, Any]] = {}
    for method in selected_methods:
        if method not in datasets_by_method:
            raise ValueError(
                f"Method '{method}' not found in loaded inputs. Available methods: "
                f"{sorted(datasets_by_method.keys())}"
            )
        _, episodes_by_road, _, resolved_followers = _prepare_method_episode_groups(
            datasets_by_method[method],
            followers=None,
            dt=None,
        )
        if road_id not in episodes_by_road:
            raise ValueError(f"Method '{method}' does not contain road {road_id}.")
        representative_episode = _select_representative_episode(
            episodes_by_road[road_id],
            follower_ids=resolved_followers,
        )
        representative_episode = {
            **representative_episode,
            "dt": float(datasets_by_method[method][0].get("dt", DEFAULT_DT)),
        }
        representative_episodes[method] = representative_episode

    availability_mask, common_dt = _boolean_union_availability(
        representative_episodes,
        agent_id=agent_id,
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    background_handle = _shade_boolean_background(
        ax,
        availability_mask,
        common_dt,
        label="hinge available",
    )
    line_handles = []
    line_labels = []
    for method in selected_methods:
        episode = representative_episodes[method]
        dt = float(episode.get("dt", common_dt))
        hinge_dis = _extract_hinge_distance_series(episode, agent_id)
        time_axis = torch.arange(hinge_dis.shape[0], dtype=torch.float32) * dt
        line = ax.plot(
            time_axis.numpy(),
            hinge_dis.numpy(),
            color=_get_method_color(method),
            linewidth=2.2,
            label=_get_method_display_name(method),
        )[0]
        line_handles.append(line)
        line_labels.append(_get_method_display_name(method))
    if background_handle is not None:
        line_handles.append(background_handle)
        line_labels.append("hinge available")
    ax.set_title(f"Road {road_id} Agent {agent_id} hinge distance comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("hinge_dis (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(line_handles, line_labels, loc="upper right", frameon=False)
    fig.tight_layout()
    hinge_dis_path = plot_dir / f"road{road_id}_agent{agent_id}_hinge_dis_comparison.png"
    fig.savefig(hinge_dis_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(hinge_dis_path)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    background_handle = _shade_boolean_background(
        ax,
        availability_mask,
        common_dt,
        label="hinge available",
    )
    line_handles = []
    line_labels = []
    for method in selected_methods:
        episode = representative_episodes[method]
        dt = float(episode.get("dt", common_dt))
        act_steer = _extract_agent_series(episode, "act_steer", agent_id)
        time_axis = torch.arange(act_steer.shape[0], dtype=torch.float32) * dt
        line = ax.plot(
            time_axis.numpy(),
            act_steer.numpy(),
            color=_get_method_color(method),
            linewidth=2.2,
            label=_get_method_display_name(method),
        )[0]
        line_handles.append(line)
        line_labels.append(_get_method_display_name(method))
    if background_handle is not None:
        line_handles.append(background_handle)
        line_labels.append("hinge available")
    ax.set_title(f"Road {road_id} Agent {agent_id} steering comparison")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering command")
    ax.grid(True, alpha=0.3)
    ax.legend(line_handles, line_labels, loc="upper right", frameon=False)
    fig.tight_layout()
    steering_path = plot_dir / f"road{road_id}_agent{agent_id}_steering_comparison.png"
    fig.savefig(steering_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(steering_path)

    return saved_paths


def resolve_input_paths(paths: Sequence[str]) -> List[Path]:
    candidate_paths = [Path(path) for path in paths] if paths else [DEFAULT_VALIDATION_RESULT_DIR]
    resolved_paths: List[Path] = []

    for candidate in candidate_paths:
        if candidate.is_dir():
            resolved_paths.extend(sorted(candidate.glob("*.pt")))
        else:
            resolved_paths.append(candidate)

    if not resolved_paths:
        raise FileNotFoundError(
            f"No .pt files found in the provided inputs: {candidate_paths}"
        )

    return resolved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read rollout_path.yaml under --data-dir and generate per-method "
            "OCCT figures and CSV tables."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/occt_comparision/action_smooth_comparision"),
        help=(
            "Directory containing rollout_path.yaml and all method result paths. "
            "Defaults to outputs/occt_comparision/action_smooth_comparision."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help=(
            "Optional method key from rollout_path.yaml. If omitted, process all methods."
        ),
    )
    parser.add_argument(
        "--compare-methods",
        nargs="+",
        default=None,
        help=(
            "Optional method keys used to merge existing per-method CSVs into "
            "comparison CSVs under --data-dir."
        ),
    )
    parser.add_argument(
        "--followers",
        nargs="+",
        type=int,
        default=None,
        help="Optional follower agent ids override.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Optional dt override. Defaults to file metadata or 0.05.",
    )
    parser.add_argument(
        "--plot-method-fill-stype",
        "--plot-method-fill-style",
        type=_str_to_bool,
        nargs="?",
        const=True,
        default=True,
        help=(
            "Whether per-method figures use fill style for agent1-3. "
            "Use true or false. Default is true."
        ),
    )
    parser.add_argument(
        "--representative-roads",
        nargs="+",
        type=int,
        default=[0],
        help="Road ids used in single-road PDF plots. Defaults to 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir not found: {data_dir}")

    method_config = _load_method_rollout_config(data_dir)
    if args.compare_methods is not None:
        selected_compare_methods = [
            _resolve_method_name(method, method_config.keys())
            for method in args.compare_methods
        ]
        comparison_csv_paths = generate_comparison_csvs(
            data_dir=data_dir,
            methods=selected_compare_methods,
        )
        for csv_path in comparison_csv_paths:
            print(f"Saved comparison CSV: {csv_path}")
        return

    if args.method is None:
        selected_methods = _ordered_methods(list(method_config.keys()))
    else:
        selected_methods = [_resolve_method_name(args.method, method_config.keys())]

    for method in selected_methods:
        source_path = _resolve_method_source_path(data_dir, method_config[method])
        output_root = _resolve_method_output_root(source_path)
        figure_dir = output_root / "figure"
        table_dir = output_root / "table"

        datasets = _load_single_method_datasets(
            method=method,
            source_path=source_path,
            followers=args.followers,
            dt=args.dt,
        )

        print(f"\nProcessing method: {_get_method_display_name(method)}")
        print(f"Source path: {source_path}")
        print(f"Figure dir: {figure_dir}")
        print(f"Table dir: {table_dir}")

        report_groups = _build_method_reports(
            datasets,
            method=method,
            followers=args.followers,
            dt=args.dt,
        )
        for report_type, rows in report_groups.items():
            _print_report_group(method, report_type, rows)
        csv_path = _write_method_csv(table_dir, method, report_groups)
        print(f"Saved CSV: {csv_path}")
        saved_plot_paths = plot_representative_road_curves(
            datasets,
            method=method,
            followers=args.followers,
            dt=args.dt,
            plot_dir=figure_dir,
            road_ids=args.representative_roads,
            plot_method_fill_stype=args.plot_method_fill_stype,
        )
        for plot_path in saved_plot_paths:
            print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
