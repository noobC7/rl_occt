import copy
import random
from typing import Any

import torch
from omegaconf import DictConfig


def _to_cpu_snapshot(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        container = [_to_cpu_snapshot(item) for item in value]
        return type(value)(container) if isinstance(value, tuple) else container
    return copy.deepcopy(value)


class FailureCurriculumBank:
    """Long-lived curriculum bank for pre-failure replay snapshots.

    The bank is trainer-owned. The environment only samples from it at reset time
    and reports episode outcomes back to the trainer.
    """

    def __init__(self, cfg: DictConfig) -> None:
        curriculum_cfg = cfg.get("curriculum", {})
        self.enabled = bool(curriculum_cfg.get("enabled", False))
        self.mode = str(curriculum_cfg.get("mode", "")).lower()
        self.is_failure_replay_mode = self.mode == "failure_precollision_replay"
        self.capacity = int(curriculum_cfg.get("bank_capacity", 4096))
        self.min_bank_size = int(curriculum_cfg.get("min_bank_size", 128))
        self.warmup_iterations = int(
            curriculum_cfg.get(
                "warmup_iterations",
                curriculum_cfg.get("warmup_epochs", 50),
            )
        )
        self.replay_prob_start = float(curriculum_cfg.get("replay_prob_start", 0.1))
        self.replay_prob_end = float(curriculum_cfg.get("replay_prob_end", 0.3))
        self.replay_prob_ramp_iterations = int(
            curriculum_cfg.get(
                "replay_prob_ramp_iterations",
                curriculum_cfg.get("replay_prob_ramp_epochs", 80),
            )
        )
        self.replace_strategy = str(
            curriculum_cfg.get("replace_strategy", "oldest_unused")
        ).lower()

        self.entries: dict[int, dict[str, Any]] = {}
        self.active_entry_ids: set[int] = set()
        self.next_entry_id = 0

        self.total_added = 0
        self.total_removed_success = 0
        self.total_replaced_capacity = 0
        self.total_updated_failure = 0
        self.total_sampled = 0
        self.total_skipped_no_slot = 0
        self.total_new_failures = 0
        self.total_replay_success = 0
        self.total_replay_failure = 0

    def __len__(self) -> int:
        return len(self.entries)

    def _choose_replacement_id(self) -> int | None:
        if not self.entries:
            return None
        candidates = [
            (entry_id, entry)
            for entry_id, entry in self.entries.items()
            if entry_id not in self.active_entry_ids
        ]
        if not candidates:
            return None

        if self.replace_strategy == "oldest_unused":
            candidates.sort(
                key=lambda item: (
                    item[1].get("last_used_iteration", -1),
                    item[1].get("created_iteration", -1),
                )
            )
            return candidates[0][0]

        return candidates[0][0]

    def _make_entry(
        self,
        *,
        entry_id: int,
        snapshot: dict[str, Any],
        failure_type: str,
        road_id: int,
        iteration: int,
    ) -> dict[str, Any]:
        return {
            "entry_id": entry_id,
            "snapshot": _to_cpu_snapshot(snapshot),
            "failure_type": str(failure_type),
            "road_id": int(road_id),
            "created_iteration": int(iteration),
            "last_used_iteration": -1,
            "times_sampled": 0,
            "times_failed_after_replay": 0,
            "times_succeeded_after_replay": 0,
        }

    def replay_enabled(self, iteration: int) -> bool:
        return (
            self.enabled
            and self.is_failure_replay_mode
            and iteration >= self.warmup_iterations
            and len(self.entries) >= self.min_bank_size
        )

    def get_replay_probability(self, iteration: int) -> float:
        if not self.replay_enabled(iteration):
            return 0.0
        if self.replay_prob_ramp_iterations <= 0:
            return self.replay_prob_end
        progress = min(
            max((iteration - self.warmup_iterations) / self.replay_prob_ramp_iterations, 0.0),
            1.0,
        )
        return (1.0 - progress) * self.replay_prob_start + progress * self.replay_prob_end

    def can_sample(self, iteration: int, road_id: int | None = None) -> bool:
        if not self.replay_enabled(iteration):
            return False
        return any(
            entry_id not in self.active_entry_ids
            and (road_id is None or entry["road_id"] == int(road_id))
            for entry_id, entry in self.entries.items()
        )

    def sample(
        self, iteration: int, road_id: int | None = None
    ) -> tuple[int, dict[str, Any]] | None:
        if not self.can_sample(iteration, road_id=road_id):
            return None
        available_entry_ids = [
            entry_id
            for entry_id, entry in self.entries.items()
            if entry_id not in self.active_entry_ids
            and (road_id is None or entry["road_id"] == int(road_id))
        ]
        if not available_entry_ids:
            return None
        entry_id = random.choice(available_entry_ids)
        entry = self.entries[entry_id]
        entry["times_sampled"] += 1
        entry["last_used_iteration"] = int(iteration)
        self.active_entry_ids.add(entry_id)
        self.total_sampled += 1
        return entry_id, _to_cpu_snapshot(entry["snapshot"])

    def add_failure(
        self,
        *,
        snapshot: dict[str, Any],
        failure_type: str,
        road_id: int,
        iteration: int,
    ) -> int | None:
        if len(self.entries) < self.capacity:
            entry_id = self.next_entry_id
            self.next_entry_id += 1
        else:
            entry_id = self._choose_replacement_id()
            if entry_id is None:
                self.total_skipped_no_slot += 1
                return None
            self.total_replaced_capacity += 1

        self.entries[entry_id] = self._make_entry(
            entry_id=entry_id,
            snapshot=snapshot,
            failure_type=failure_type,
            road_id=road_id,
            iteration=iteration,
        )
        self.total_added += 1
        return entry_id

    def remove(self, entry_id: int) -> None:
        self.entries.pop(entry_id, None)
        self.active_entry_ids.discard(entry_id)

    def update_after_replay_failure(
        self,
        *,
        entry_id: int,
        snapshot: dict[str, Any] | None,
        failure_type: str,
        road_id: int,
        iteration: int,
    ) -> None:
        self.active_entry_ids.discard(entry_id)
        if entry_id not in self.entries:
            return

        if snapshot is None:
            self.entries[entry_id]["times_failed_after_replay"] += 1
            return

        previous_entry = self.entries[entry_id]
        self.entries[entry_id] = self._make_entry(
            entry_id=entry_id,
            snapshot=snapshot,
            failure_type=failure_type,
            road_id=road_id,
            iteration=iteration,
        )
        self.entries[entry_id]["times_sampled"] = previous_entry["times_sampled"]
        self.entries[entry_id]["last_used_iteration"] = previous_entry["last_used_iteration"]
        self.entries[entry_id]["times_failed_after_replay"] = (
            previous_entry["times_failed_after_replay"] + 1
        )
        self.entries[entry_id]["times_succeeded_after_replay"] = previous_entry[
            "times_succeeded_after_replay"
        ]
        self.total_updated_failure += 1

    def resolve_event(self, event: dict[str, Any], iteration: int) -> None:
        event_type = str(event.get("event_type", "")).lower()
        if event_type == "new_failure":
            self.total_new_failures += 1
            snapshot = event.get("snapshot", None)
            if snapshot is None:
                return
            self.add_failure(
                snapshot=snapshot,
                failure_type=str(event.get("failure_type", "unknown")),
                road_id=int(event.get("road_id", -1)),
                iteration=iteration,
            )
            return

        entry_id = int(event.get("source_entry_id", -1))
        if entry_id < 0:
            return

        if event_type == "replay_success":
            self.total_replay_success += 1
            if entry_id in self.entries:
                self.entries[entry_id]["times_succeeded_after_replay"] += 1
            self.remove(entry_id)
            self.total_removed_success += 1
            return

        if event_type == "replay_failure":
            self.total_replay_failure += 1
            self.update_after_replay_failure(
                entry_id=entry_id,
                snapshot=event.get("snapshot", None),
                failure_type=str(event.get("failure_type", "unknown")),
                road_id=int(event.get("road_id", -1)),
                iteration=iteration,
            )
            return

        self.active_entry_ids.discard(entry_id)

    def metrics(self, iteration: int) -> dict[str, float]:
        return {
            "train/failure_curriculum/enabled": float(
                self.enabled and self.is_failure_replay_mode
            ),
            "train/failure_curriculum/bank_size": float(len(self.entries)),
            "train/failure_curriculum/active_replays": float(len(self.active_entry_ids)),
            "train/failure_curriculum/replay_probability": float(
                self.get_replay_probability(iteration)
            ),
            "train/failure_curriculum/total_added": float(self.total_added),
            "train/failure_curriculum/total_removed_success": float(
                self.total_removed_success
            ),
            "train/failure_curriculum/total_updated_failure": float(
                self.total_updated_failure
            ),
            "train/failure_curriculum/total_sampled": float(self.total_sampled),
            "train/failure_curriculum/total_skipped_no_slot": float(
                self.total_skipped_no_slot
            ),
        }
