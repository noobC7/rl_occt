import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from occt_metrics_evaluation import get_valid_length


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


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float32).mean().item())


def _safe_std(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(values, dtype=torch.float32).std(unbiased=False).item())


def analyze_hinge_transitions(
    rollout_path: Path,
    *,
    follower_ids: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rollouts = torch.load(rollout_path, map_location="cpu", weights_only=False)
    trajectories = list(rollouts.unbind(0))

    event_rows: list[dict[str, Any]] = []
    per_road_values: dict[int, dict[str, list[float]]] = {}
    overall_values = {
        "steering_jump_abs": [],
        "acc_jump_abs": [],
    }

    for episode_index, traj in enumerate(trajectories):
        valid_len = get_valid_length(traj)
        info = traj["next", "agents", "info"][:valid_len]
        road_id = int(info["road_batch_id"].reshape(-1)[-1].item())
        per_road_values.setdefault(
            road_id,
            {"steering_jump_abs": [], "acc_jump_abs": []},
        )

        hinge_status = info["hinge_status"].float().squeeze(-1).cpu()
        act_steer = info["act_steer"].float().squeeze(-1).cpu()
        command_acc = info["command_acceleration"].float().cpu()

        steering_delta = torch.zeros_like(act_steer)
        steering_delta[1:] = act_steer[1:] - act_steer[:-1]
        acc_delta = torch.zeros_like(command_acc)
        acc_delta[1:] = command_acc[1:] - command_acc[:-1]

        for agent_id in follower_ids:
            if agent_id >= hinge_status.shape[1]:
                continue
            status_delta = torch.zeros(valid_len, dtype=torch.float32)
            if valid_len > 1:
                status_delta[1:] = hinge_status[1:, agent_id] - hinge_status[:-1, agent_id]
            transition_steps = torch.nonzero(status_delta.abs() > 0.5, as_tuple=True)[0]

            for step_index in transition_steps.tolist():
                steer_jump_abs = float(abs(steering_delta[step_index, agent_id].item()))
                acc_jump_abs = float(abs(acc_delta[step_index, agent_id].item()))
                transition_direction = "0_to_1" if status_delta[step_index].item() > 0 else "1_to_0"

                event_rows.append(
                    {
                        "road_id": road_id,
                        "episode_index": episode_index,
                        "agent_id": agent_id,
                        "step_index": step_index,
                        "transition_direction": transition_direction,
                        "hinge_status_prev": float(hinge_status[step_index - 1, agent_id].item())
                        if step_index > 0
                        else float("nan"),
                        "hinge_status_curr": float(hinge_status[step_index, agent_id].item()),
                        "steering_jump_abs": steer_jump_abs,
                        "acc_jump_abs": acc_jump_abs,
                    }
                )
                per_road_values[road_id]["steering_jump_abs"].append(steer_jump_abs)
                per_road_values[road_id]["acc_jump_abs"].append(acc_jump_abs)
                overall_values["steering_jump_abs"].append(steer_jump_abs)
                overall_values["acc_jump_abs"].append(acc_jump_abs)

    per_road_rows = []
    for road_id in sorted(per_road_values):
        road_stats = per_road_values[road_id]
        per_road_rows.append(
            {
                "road_id": road_id,
                "transition_count": len(road_stats["steering_jump_abs"]),
                "steering_jump_abs_mean": _safe_mean(road_stats["steering_jump_abs"]),
                "steering_jump_abs_std": _safe_std(road_stats["steering_jump_abs"]),
                "steering_jump_abs_max": max(road_stats["steering_jump_abs"])
                if road_stats["steering_jump_abs"]
                else float("nan"),
                "acc_jump_abs_mean": _safe_mean(road_stats["acc_jump_abs"]),
                "acc_jump_abs_std": _safe_std(road_stats["acc_jump_abs"]),
                "acc_jump_abs_max": max(road_stats["acc_jump_abs"])
                if road_stats["acc_jump_abs"]
                else float("nan"),
            }
        )

    overall_rows = [
        {
            "transition_count": len(overall_values["steering_jump_abs"]),
            "steering_jump_abs_mean": _safe_mean(overall_values["steering_jump_abs"]),
            "steering_jump_abs_std": _safe_std(overall_values["steering_jump_abs"]),
            "steering_jump_abs_max": max(overall_values["steering_jump_abs"])
            if overall_values["steering_jump_abs"]
            else float("nan"),
            "acc_jump_abs_mean": _safe_mean(overall_values["acc_jump_abs"]),
            "acc_jump_abs_std": _safe_std(overall_values["acc_jump_abs"]),
            "acc_jump_abs_max": max(overall_values["acc_jump_abs"])
            if overall_values["acc_jump_abs"]
            else float("nan"),
        }
    ]

    return event_rows, per_road_rows, overall_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze action jumps exactly at hinge_status transition steps."
    )
    parser.add_argument("rollout_path", type=Path, help="Path to rollout .pt file.")
    parser.add_argument(
        "--follower-ids",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Follower agent ids to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "hinge_transition_reports",
        help="Directory used to save CSV reports.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_rows, per_road_rows, overall_rows = analyze_hinge_transitions(
        args.rollout_path,
        follower_ids=args.follower_ids,
    )

    output_dir = args.output_dir
    events_path = output_dir / "hinge_transition_events.csv"
    per_road_path = output_dir / "hinge_transition_per_road.csv"
    overall_path = output_dir / "hinge_transition_overall.csv"

    _write_csv(events_path, event_rows)
    _write_csv(per_road_path, per_road_rows)
    _write_csv(overall_path, overall_rows)

    print(f"Saved event CSV: {events_path}")
    print(f"Saved per-road CSV: {per_road_path}")
    print(f"Saved overall CSV: {overall_path}")

    print("\nPer-road summary:")
    for row in per_road_rows:
        print(row)

    print("\nOverall summary:")
    for row in overall_rows:
        print(row)


if __name__ == "__main__":
    main()
