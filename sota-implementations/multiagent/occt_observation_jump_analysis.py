import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from occt_metrics_evaluation import get_valid_length


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "mappo_mlp_continues_act_change_penalty.yaml"
)
DEFAULT_ROLLOUT_PATH = Path(
    "/home/yons/Graduation/rl_occt/outputs/occt_comparision/action_smooth_comparision/"
    "13-31-45_mlp_continuous_act_change_penalty_history10_eval/"
    "run-20260327_133148-shns94k7251nkjfbhybge/rollouts/"
    "rollout_iter_0_frames_0_paths_0_5.pt"
)
DEFAULT_OUTPUT_DIR = Path.cwd() / "observation_jump_reports"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def _build_flat_observation_layout(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    scenario_cfg = cfg["env"]["scenario"]
    task_class = int(scenario_cfg.get("task_class", 1))
    mask_ref_v = bool(scenario_cfg.get("mask_ref_v", False))
    n_points_short_term = int(scenario_cfg.get("n_points_short_term", 4))
    n_nearing_agents = int(scenario_cfg.get("n_nearing_agents_observed", 2))
    n_observed_steps = int(scenario_cfg.get("n_observed_steps", 1))
    n_points_nearing_boundary = int(
        scenario_cfg.get("n_points_nearing_boundary", n_points_short_term + 1)
    )
    self_boundary_distance_dim = max(0, n_points_nearing_boundary - 1)

    layout_spec = [
        ("self_vel", 1),
        ("self_speed", 1),
        ("self_steering", 1),
        ("self_acc", 1),
        ("self_ref_velocity", 0 if mask_ref_v else n_points_short_term),
        ("self_ref_points", n_points_short_term * 2),
    ]

    if task_class == 1:
        layout_spec.extend(
            [
                ("self_hinge_velocity", 0 if mask_ref_v else n_points_short_term),
                ("self_hinge_points", n_points_short_term * 2),
                ("self_left_boundary_distance", self_boundary_distance_dim),
                ("self_right_boundary_distance", self_boundary_distance_dim),
                ("self_hinge_status", n_points_short_term),
                ("self_distance_to_ref", 1),
                ("self_distance_to_hinge", 1),
                ("self_distance_to_left_boundary", 1),
                ("self_distance_to_right_boundary", 1),
                ("self_error_vel", 2),
                ("self_error_space", 2),
            ]
        )
    else:
        layout_spec.extend(
            [
                ("self_distance_to_ref", 1),
                ("self_distance_to_left_boundary", 1),
                ("self_distance_to_right_boundary", 1),
                ("self_error_vel", 2),
                ("self_error_space", 2),
            ]
        )

    layout = []
    cursor = 0
    for name, dim in layout_spec:
        if dim <= 0:
            continue
        layout.append(
            {
                "name": name,
                "segments": [(cursor, cursor + dim)],
                "dim": dim,
            }
        )
        cursor += dim

    other_feat_spec = [
        ("others_pos", 2),
        ("others_rot", 1),
        ("others_relative_longitudinal_velocity", 1),
        ("others_relative_acceleration", n_observed_steps),
        ("others_distance", 1),
    ]
    other_segments = {name: [] for name, _ in other_feat_spec}
    for _ in range(n_nearing_agents):
        for name, dim in other_feat_spec:
            other_segments[name].append((cursor, cursor + dim))
            cursor += dim
    for name, dim in other_feat_spec:
        layout.append(
            {
                "name": name,
                "segments": other_segments[name],
                "dim": dim * n_nearing_agents,
            }
        )
    return layout


def _extract_group_tensor(obs: torch.Tensor, layout_item: dict[str, Any]) -> torch.Tensor:
    segments = layout_item["segments"]
    parts = [obs[..., start:end] for start, end in segments]
    if len(parts) == 1:
        return parts[0]
    return torch.cat(parts, dim=-1)


def _tensor_to_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _safe_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return _tensor_to_float(torch.quantile(values, q))


def analyze_rollout(
    rollout_path: Path,
    layout: list[dict[str, Any]],
    *,
    agent_ids: list[int] | None,
    top_k_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rollouts = torch.load(rollout_path, map_location="cpu", weights_only=False)
    trajectories = list(rollouts.unbind(0))
    summary_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    total_obs_dims = (
        max(end for item in layout for _, end in item["segments"]) if layout else 0
    )
    total_steps = 0
    total_events = 0

    for layout_item in layout:
        name = layout_item["name"]
        group_dim = layout_item["dim"]
        abs_deltas = []
        max_events = []
        steer_aligned_deltas = []

        for episode_index, traj in enumerate(trajectories):
            valid_len = get_valid_length(traj)
            obs = traj["agents", "observation"][:valid_len].detach().cpu().float()
            if obs.ndim < 3:
                raise ValueError(
                    f"Expected observation tensor with shape [T, ..., n_agents, obs_dim], got {tuple(obs.shape)}."
                )
            group_slice_end = max(end for _, end in layout_item["segments"])
            if obs.shape[-1] < group_slice_end:
                raise ValueError(
                    f"Observation dim {obs.shape[-1]} is smaller than expected slice end {group_slice_end} "
                    f"for group '{name}'."
                )
            n_agents = obs.shape[-2]
            resolved_agent_ids = (
                [agent_id for agent_id in agent_ids if 0 <= agent_id < n_agents]
                if agent_ids is not None
                else list(range(n_agents))
            )
            if not resolved_agent_ids:
                continue

            act_steer = traj["next", "agents", "info", "act_steer"][:valid_len].detach().cpu().float()
            if act_steer.ndim > 2:
                act_steer = act_steer.squeeze(-1)
            steer_delta_abs = torch.zeros_like(act_steer)
            if valid_len > 1:
                steer_delta_abs[1:] = (act_steer[1:] - act_steer[:-1]).abs()

            obs_group = _extract_group_tensor(obs, layout_item)
            obs_delta = torch.zeros_like(obs_group)
            if valid_len > 1:
                obs_delta[1:] = obs_group[1:] - obs_group[:-1]
            abs_obs_delta = obs_delta.abs()

            for agent_id in resolved_agent_ids:
                agent_delta = abs_obs_delta[:, agent_id].reshape(valid_len, -1)
                abs_deltas.append(agent_delta.reshape(-1))

                per_step_group_max = agent_delta.max(dim=-1).values
                steer_group_pair = torch.stack(
                    (steer_delta_abs[:, agent_id], per_step_group_max), dim=-1
                )
                steer_aligned_deltas.append(steer_group_pair)

                if valid_len <= 1:
                    continue
                top_values, top_indices = torch.topk(
                    per_step_group_max[1:],
                    k=min(top_k_events, per_step_group_max[1:].numel()),
                )
                for value, local_index in zip(top_values, top_indices):
                    step_index = int(local_index.item()) + 1
                    max_events.append(
                        {
                            "group_name": name,
                            "episode_index": episode_index,
                            "agent_id": agent_id,
                            "step_index": step_index,
                            "obs_delta_max_abs": _tensor_to_float(value),
                            "steer_delta_abs": _tensor_to_float(
                                steer_delta_abs[step_index, agent_id]
                            ),
                            "group_dim": group_dim,
                        }
                    )

        if not abs_deltas:
            continue

        stacked_abs_delta = torch.cat(abs_deltas)
        steer_alignment = torch.cat(steer_aligned_deltas, dim=0)
        summary_rows.append(
            {
                "group_name": name,
                "group_dim": group_dim,
                "slice_start": ",".join(str(start) for start, _ in layout_item["segments"]),
                "slice_end": ",".join(str(end) for _, end in layout_item["segments"]),
                "max_abs_delta": _tensor_to_float(stacked_abs_delta.max()),
                "mean_abs_delta": _tensor_to_float(stacked_abs_delta.mean()),
                "p95_abs_delta": _safe_quantile(stacked_abs_delta, 0.95),
                "p99_abs_delta": _safe_quantile(stacked_abs_delta, 0.99),
                "mean_step_max_abs_delta": _tensor_to_float(
                    steer_alignment[:, 1].mean()
                ),
                "max_step_max_abs_delta": _tensor_to_float(steer_alignment[:, 1].max()),
                "mean_abs_delta_at_steer_jump": _tensor_to_float(
                    steer_alignment[steer_alignment[:, 0] > 0.05, 1].mean()
                )
                if bool((steer_alignment[:, 0] > 0.05).any().item())
                else float("nan"),
            }
        )
        max_events = sorted(
            max_events, key=lambda item: item["obs_delta_max_abs"], reverse=True
        )
        event_rows.extend(max_events[:top_k_events])

    summary_rows = sorted(
        summary_rows, key=lambda item: item["max_step_max_abs_delta"], reverse=True
    )
    event_rows = sorted(
        event_rows, key=lambda item: item["obs_delta_max_abs"], reverse=True
    )

    for traj in trajectories:
        valid_len = get_valid_length(traj)
        total_steps += valid_len
        total_events += max(valid_len - 1, 0)

    metadata = {
        "rollout_path": str(rollout_path),
        "episodes": len(trajectories),
        "total_valid_steps": total_steps,
        "total_delta_events": total_events,
        "observation_dim_expected": total_obs_dims,
    }
    return summary_rows, event_rows, metadata


def analyze_steering_jump_context(
    rollout_path: Path,
    layout: list[dict[str, Any]],
    *,
    agent_ids: list[int] | None,
    top_k_steering_events: int,
    top_n_groups_per_event: int,
) -> list[dict[str, Any]]:
    rollouts = torch.load(rollout_path, map_location="cpu", weights_only=False)
    trajectories = list(rollouts.unbind(0))
    steering_event_candidates: list[dict[str, Any]] = []
    per_agent_group_delta: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    for episode_index, traj in enumerate(trajectories):
        valid_len = get_valid_length(traj)
        obs = traj["agents", "observation"][:valid_len].detach().cpu().float()
        act_steer = traj["next", "agents", "info", "act_steer"][:valid_len].detach().cpu().float()
        if act_steer.ndim > 2:
            act_steer = act_steer.squeeze(-1)
        steer_delta_abs = torch.zeros_like(act_steer)
        if valid_len > 1:
            steer_delta_abs[1:] = (act_steer[1:] - act_steer[:-1]).abs()

        n_agents = obs.shape[-2]
        resolved_agent_ids = (
            [agent_id for agent_id in agent_ids if 0 <= agent_id < n_agents]
            if agent_ids is not None
            else list(range(n_agents))
        )
        for agent_id in resolved_agent_ids:
            group_delta_map: dict[str, torch.Tensor] = {}
            for layout_item in layout:
                obs_group = _extract_group_tensor(obs, layout_item)
                obs_delta = torch.zeros_like(obs_group[:, agent_id])
                if valid_len > 1:
                    obs_delta[1:] = obs_group[1:, agent_id] - obs_group[:-1, agent_id]
                group_delta_map[layout_item["name"]] = obs_delta.abs().reshape(valid_len, -1).max(dim=-1).values
            per_agent_group_delta[(episode_index, agent_id)] = group_delta_map

            if valid_len <= 1:
                continue
            top_values, top_indices = torch.topk(
                steer_delta_abs[1:, agent_id],
                k=min(top_k_steering_events, steer_delta_abs[1:, agent_id].numel()),
            )
            for value, local_index in zip(top_values, top_indices):
                steering_event_candidates.append(
                    {
                        "episode_index": episode_index,
                        "agent_id": agent_id,
                        "step_index": int(local_index.item()) + 1,
                        "steer_delta_abs": _tensor_to_float(value),
                    }
                )

    steering_event_candidates = sorted(
        steering_event_candidates, key=lambda item: item["steer_delta_abs"], reverse=True
    )[:top_k_steering_events]

    detailed_rows: list[dict[str, Any]] = []
    for event_rank, event in enumerate(steering_event_candidates, start=1):
        episode_index = event["episode_index"]
        agent_id = event["agent_id"]
        step_index = event["step_index"]
        group_delta_map = per_agent_group_delta[(episode_index, agent_id)]
        ranked_groups = sorted(
            (
                {
                    "group_name": group_name,
                    "obs_delta_max_abs": _tensor_to_float(group_values[step_index]),
                }
                for group_name, group_values in group_delta_map.items()
            ),
            key=lambda item: item["obs_delta_max_abs"],
            reverse=True,
        )[:top_n_groups_per_event]
        for group_rank, group_item in enumerate(ranked_groups, start=1):
            detailed_rows.append(
                {
                    "event_rank": event_rank,
                    "episode_index": episode_index,
                    "agent_id": agent_id,
                    "step_index": step_index,
                    "steer_delta_abs": event["steer_delta_abs"],
                    "group_rank": group_rank,
                    "group_name": group_item["group_name"],
                    "obs_delta_max_abs": group_item["obs_delta_max_abs"],
                }
            )
    return detailed_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["empty"])
        return
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_top_summary(summary_rows: list[dict[str, Any]], top_n: int = 10) -> None:
    print("\nTop observation groups by max step delta:")
    for row in summary_rows[:top_n]:
        print(
            f"{row['group_name']:35s} "
            f"max_step={row['max_step_max_abs_delta']:.4f} "
            f"p99={row['p99_abs_delta']:.4f} "
            f"mean={row['mean_abs_delta']:.4f}"
        )


def _print_top_events(event_rows: list[dict[str, Any]], top_n: int = 15) -> None:
    print("\nTop observation jump events:")
    for row in event_rows[:top_n]:
        print(
            f"{row['group_name']:35s} "
            f"ep={row['episode_index']} "
            f"agent={row['agent_id']} "
            f"step={row['step_index']} "
            f"obs_jump={row['obs_delta_max_abs']:.4f} "
            f"steer_jump={row['steer_delta_abs']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which observation groups exhibit the largest numerical jumps "
            "in OCCT rollout data."
        )
    )
    parser.add_argument(
        "rollout_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ROLLOUT_PATH,
        help="Path to rollout .pt file.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training config used to build the observation layout.",
    )
    parser.add_argument(
        "--agent-ids",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Agent ids to analyze. Defaults to follower agents 1 2 3.",
    )
    parser.add_argument(
        "--top-k-events",
        type=int,
        default=20,
        help="Number of top jump events to retain per group before global sorting.",
    )
    parser.add_argument(
        "--top-k-steering-events",
        type=int,
        default=10,
        help="Number of largest steering jump events to analyze in detail.",
    )
    parser.add_argument(
        "--top-n-groups-per-event",
        type=int,
        default=8,
        help="Number of observation groups to retain for each steering jump event.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save CSV/JSON reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config_path)
    layout = _build_flat_observation_layout(cfg)
    summary_rows, event_rows, metadata = analyze_rollout(
        args.rollout_path,
        layout,
        agent_ids=args.agent_ids,
        top_k_events=args.top_k_events,
    )
    steering_context_rows = analyze_steering_jump_context(
        args.rollout_path,
        layout,
        agent_ids=args.agent_ids,
        top_k_steering_events=args.top_k_steering_events,
        top_n_groups_per_event=args.top_n_groups_per_event,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "observation_group_jump_summary.csv"
    events_path = output_dir / "observation_jump_events.csv"
    steering_context_path = output_dir / "steering_jump_context.csv"
    metadata_path = output_dir / "observation_jump_metadata.json"

    _write_csv(summary_path, summary_rows)
    _write_csv(events_path, event_rows)
    _write_csv(steering_context_path, steering_context_rows)
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved event CSV: {events_path}")
    print(f"Saved steering-context CSV: {steering_context_path}")
    print(f"Saved metadata JSON: {metadata_path}")
    _print_top_summary(summary_rows)
    _print_top_events(event_rows)


if __name__ == "__main__":
    main()
