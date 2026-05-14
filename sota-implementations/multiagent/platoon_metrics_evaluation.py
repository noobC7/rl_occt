import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch


DEFAULT_DT = 0.05
DEFAULT_WHEELBASE_M = 2.32
DEFAULT_IPPO_ROLLOUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "platoon_comparison_0411"
    / "22-19-18_platoon_ippo_eval"
    / "run-20260413_221924-cfwz0f7yugprulfwzm16t"
    / "rollouts"
    / "rollout_iter_100_frames_6060000_paths_0_1.pt"
)
DEFAULT_MAPPO_ROLLOUT = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "platoon_comparison_0411"
    / "22-28-21_platoon_mappo_eval"
    / "run-20260413_222833-x7uviscrmexv89wavl3fk"
    / "rollouts"
    / "rollout_iter_100_frames_6060000_paths_0_1.pt"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs"
    / "platoon_comparison_0411"
    / "platoon_metrics_tables"
)

SCENE_SPECS = (
    ("right_angle_turn", [0], "Right-angle turn (batch 0)"),
    ("straight", [1], "Straight (batch 1)"),
    ("overall", [0, 1], "Overall (batches 0 + 1)"),
)
METHOD_PATHS = {
    "ippo": DEFAULT_IPPO_ROLLOUT,
    "mappo": DEFAULT_MAPPO_ROLLOUT,
}
METRIC_ORDER = (
    "average_speed",
    "speed_error",
    "spacing_error",
    "lateral_deviation",
    "jerk",
    "lateral_accel",
)
METRIC_LABELS = {
    "average_speed": "average_speed_mps",
    "speed_error": "speed_error_mps",
    "spacing_error": "spacing_error_m",
    "lateral_deviation": "lateral_deviation_m",
    "jerk": "jerk_mps3",
    "lateral_accel": "lateral_accel_mps2",
}
STAT_ORDER = ("mean", "min", "max")


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim > 0 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1)
    return tensor


def _get_valid_length(traj) -> int:
    next_done = traj.get(("next", "done")).sum(
        tuple(range(traj.batch_dims, traj.get(("next", "done")).ndim)),
        dtype=torch.bool,
    )
    if next_done.any():
        return int(next_done.nonzero(as_tuple=True)[0][0].item()) + 1
    return int(traj.batch_size[0])


def _compute_jerk_from_acceleration(acceleration: torch.Tensor, dt: float) -> torch.Tensor:
    acceleration = torch.as_tensor(acceleration, dtype=torch.float32)
    jerk = torch.zeros_like(acceleration)
    if acceleration.shape[0] > 1:
        jerk[1:] = (acceleration[1:] - acceleration[:-1]) / max(dt, 1e-6)
    return jerk.abs()


def _extract_speed_error_samples(info, valid_len: int) -> torch.Tensor:
    if "platoon_error_vel" in info:
        value = _squeeze_trailing_unit_dim(info["platoon_error_vel"][:valid_len]).float()
        if value.ndim >= 3:
            value = value[..., 0]
        return value.abs().reshape(-1)

    vel_norm = _squeeze_trailing_unit_dim(info["vel_norm"][:valid_len]).float()
    ref_vel = _squeeze_trailing_unit_dim(info["ref_vel"][:valid_len]).float()
    return (vel_norm - ref_vel).abs().reshape(-1)


def _extract_average_speed_samples(info, valid_len: int) -> torch.Tensor:
    if "vel_norm" not in info:
        raise KeyError("Missing 'vel_norm' in rollout info.")

    speed = _squeeze_trailing_unit_dim(info["vel_norm"][:valid_len]).float()
    if speed.ndim <= 1:
        return speed.reshape(-1)

    agent_dims = tuple(range(1, speed.ndim))
    return speed.mean(dim=agent_dims).reshape(-1)


def _extract_spacing_error_samples(info, valid_len: int) -> torch.Tensor:
    if "error_space" not in info:
        raise KeyError("Missing 'error_space' in rollout info.")

    value = _squeeze_trailing_unit_dim(info["error_space"][:valid_len]).float()
    if value.ndim != 3 or value.shape[-1] != 2:
        raise ValueError(
            f"Expected 'error_space' with shape [time, agents, 2], got {tuple(value.shape)}."
        )

    num_agents = value.shape[1]
    if num_agents == 1:
        return value[..., 0].abs().reshape(-1)

    # Keep one spacing-error sample per vehicle per timestep:
    # the leader uses the rear spacing error, and all remaining vehicles
    # use the front spacing error to the predecessor.
    per_agent_error = torch.empty(
        value.shape[:2],
        device=value.device,
        dtype=value.dtype,
    )
    per_agent_error[:, 0] = value[:, 0, 1]
    per_agent_error[:, 1:] = value[:, 1:, 0]
    return per_agent_error.abs().reshape(-1)


def _extract_lateral_deviation_samples(info, valid_len: int) -> torch.Tensor:
    if "distance_ref" not in info:
        raise KeyError("Missing 'distance_ref' in rollout info.")
    return _squeeze_trailing_unit_dim(info["distance_ref"][:valid_len]).float().abs().reshape(-1)


def _extract_jerk_samples(info, valid_len: int, dt: float) -> torch.Tensor:
    if "command_jerk_abs" in info:
        return _squeeze_trailing_unit_dim(info["command_jerk_abs"][:valid_len]).float().reshape(-1)
    if "command_jerk" in info:
        return _squeeze_trailing_unit_dim(info["command_jerk"][:valid_len]).float().abs().reshape(-1)
    if "act_acc" not in info:
        raise KeyError("Missing both jerk fields and 'act_acc' in rollout info.")
    act_acc = _squeeze_trailing_unit_dim(info["act_acc"][:valid_len]).float()
    return _compute_jerk_from_acceleration(act_acc, dt).reshape(-1)


def _extract_lateral_accel_samples(info, valid_len: int, wheelbase_m: float) -> torch.Tensor:
    if "vel_norm" not in info or "act_steer" not in info:
        raise KeyError("Missing 'vel_norm' or 'act_steer' in rollout info.")
    speed = _squeeze_trailing_unit_dim(info["vel_norm"][:valid_len]).float()
    steering = _squeeze_trailing_unit_dim(info["act_steer"][:valid_len]).float()
    lateral_accel = speed.square() * torch.abs(torch.tan(steering)) / max(wheelbase_m, 1e-6)
    return lateral_accel.reshape(-1)


def _extract_episode_metric_samples(
    traj,
    *,
    dt: float,
    wheelbase_m: float,
) -> Dict[str, torch.Tensor]:
    valid_len = _get_valid_length(traj)
    info = traj["next"]["agents"]["info"]
    return {
        "average_speed": _extract_average_speed_samples(info, valid_len),
        "speed_error": _extract_speed_error_samples(info, valid_len),
        "spacing_error": _extract_spacing_error_samples(info, valid_len),
        "lateral_deviation": _extract_lateral_deviation_samples(info, valid_len),
        "jerk": _extract_jerk_samples(info, valid_len, dt),
        "lateral_accel": _extract_lateral_accel_samples(info, valid_len, wheelbase_m),
    }


def _cat_tensors(chunks: Iterable[torch.Tensor]) -> torch.Tensor:
    tensors = [chunk.reshape(-1).detach().cpu().float() for chunk in chunks if chunk.numel() > 0]
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors, dim=0)


def _compute_stats(samples: torch.Tensor) -> Dict[str, float]:
    if samples.numel() == 0:
        return {stat_name: float("nan") for stat_name in STAT_ORDER}
    return {
        "mean": float(samples.mean().item()),
        "min": float(samples.min().item()),
        "max": float(samples.max().item()),
    }


def compute_scene_table(
    rollout_path: Path,
    *,
    method_name: str,
    batch_indices: Sequence[int],
    dt: float,
    wheelbase_m: float,
) -> Dict[str, float]:
    rollouts = _torch_load(rollout_path)
    trajectories = list(rollouts.unbind(0))
    metric_chunks = {metric_name: [] for metric_name in METRIC_ORDER}

    for batch_idx in batch_indices:
        if batch_idx < 0 or batch_idx >= len(trajectories):
            raise IndexError(
                f"Batch index {batch_idx} is out of range for {rollout_path} "
                f"(available batches: 0..{len(trajectories) - 1})."
            )
        episode_samples = _extract_episode_metric_samples(
            trajectories[batch_idx],
            dt=dt,
            wheelbase_m=wheelbase_m,
        )
        for metric_name, samples in episode_samples.items():
            metric_chunks[metric_name].append(samples)

    row: Dict[str, float] = {"method": method_name}
    for metric_name in METRIC_ORDER:
        stats = _compute_stats(_cat_tensors(metric_chunks[metric_name]))
        for stat_name in STAT_ORDER:
            row[f"{metric_name}_{stat_name}"] = stats[stat_name]
    return row


def build_scene_rows(
    *,
    method_paths: Dict[str, Path],
    batch_indices: Sequence[int],
    dt: float,
    wheelbase_m: float,
) -> List[Dict[str, float]]:
    rows = []
    for method_name, rollout_path in method_paths.items():
        rows.append(
            compute_scene_table(
                rollout_path,
                method_name=method_name,
                batch_indices=batch_indices,
                dt=dt,
                wheelbase_m=wheelbase_m,
            )
        )
    return rows


def _format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.6f}"


def _table_fieldnames() -> List[str]:
    fieldnames = ["method"]
    for metric_name in METRIC_ORDER:
        for stat_name in STAT_ORDER:
            fieldnames.append(f"{metric_name}_{stat_name}")
    return fieldnames


def write_csv(output_path: Path, rows: Sequence[Dict[str, float]]) -> None:
    fieldnames = _table_fieldnames()
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown_table(title: str, rows: Sequence[Dict[str, float]]) -> str:
    columns = ["method"]
    for metric_name in METRIC_ORDER:
        metric_label = METRIC_LABELS[metric_name]
        for stat_name in STAT_ORDER:
            columns.append(f"{metric_label}_{stat_name}")

    markdown_lines = [f"## {title}"]
    markdown_lines.append("| " + " | ".join(columns) + " |")
    markdown_lines.append("| " + " | ".join(["---"] + ["---:"] * (len(columns) - 1)) + " |")

    for row in rows:
        cells = [str(row["method"])]
        for metric_name in METRIC_ORDER:
            for stat_name in STAT_ORDER:
                cells.append(_format_float(row[f"{metric_name}_{stat_name}"]))
        markdown_lines.append("| " + " | ".join(cells) + " |")
    markdown_lines.append("")
    return "\n".join(markdown_lines)


def write_markdown(output_path: Path, markdown_sections: Sequence[str]) -> None:
    output_path.write_text("\n".join(markdown_sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare platoon metrics between IPPO and MAPPO rollouts and generate "
            "three tables for right-angle turn, straight, and overall scenes."
        )
    )
    parser.add_argument(
        "--ippo-rollout",
        type=Path,
        default=DEFAULT_IPPO_ROLLOUT,
        help="Path to the IPPO rollout .pt file.",
    )
    parser.add_argument(
        "--mappo-rollout",
        type=Path,
        default=DEFAULT_MAPPO_ROLLOUT,
        help="Path to the MAPPO rollout .pt file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save CSV and Markdown tables.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Simulation timestep in seconds.",
    )
    parser.add_argument(
        "--wheelbase-m",
        type=float,
        default=DEFAULT_WHEELBASE_M,
        help="Wheelbase used to compute lateral acceleration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_paths = {
        "ippo": args.ippo_rollout.expanduser().resolve(),
        "mappo": args.mappo_rollout.expanduser().resolve(),
    }
    for method_name, rollout_path in method_paths.items():
        if not rollout_path.exists():
            raise FileNotFoundError(f"{method_name} rollout not found: {rollout_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_sections: List[str] = []
    saved_csv_paths: List[Path] = []

    for scene_name, batch_indices, title in SCENE_SPECS:
        rows = build_scene_rows(
            method_paths=method_paths,
            batch_indices=batch_indices,
            dt=args.dt,
            wheelbase_m=args.wheelbase_m,
        )
        csv_path = output_dir / f"{scene_name}_comparison.csv"
        write_csv(csv_path, rows)
        saved_csv_paths.append(csv_path)
        markdown_sections.append(render_markdown_table(title, rows))

    markdown_path = output_dir / "platoon_metrics_tables.md"
    write_markdown(markdown_path, markdown_sections)

    print(f"Saved markdown table summary: {markdown_path}")
    for csv_path in saved_csv_paths:
        print(f"Saved CSV: {csv_path}")
    print()
    for section in markdown_sections:
        print(section)


if __name__ == "__main__":
    main()
