from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "sota-implementations"
    / "multiagent"
    / "occt_metrics_evaluation.py"
)


def _load_occt_metrics_module():
    sys.path.insert(0, str(MODULE_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "occt_metrics_evaluation",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_episode(front_errors: list[float], lateral_errors: list[float]):
    steps = len(front_errors)
    assert steps == len(lateral_errors)

    num_agents = 3
    error_space = torch.zeros(steps, num_agents, 2, dtype=torch.float32)
    error_space[:, 1, 0] = torch.tensor(front_errors, dtype=torch.float32)
    error_space[:, 2, 0] = torch.tensor(front_errors, dtype=torch.float32)

    distance_ref = torch.zeros(steps, num_agents, dtype=torch.float32)
    distance_ref[:, 1] = torch.tensor(lateral_errors, dtype=torch.float32)
    distance_ref[:, 2] = torch.tensor(lateral_errors, dtype=torch.float32)

    bool_info = torch.zeros(steps, num_agents, dtype=torch.bool)
    float_info = torch.zeros(steps, num_agents, dtype=torch.float32)
    vel = torch.zeros(steps, num_agents, 2, dtype=torch.float32)

    return {
        "num_steps": steps,
        "road_id": 4,
        "road_name": "s_curve",
        "info": {
            "error_space": error_space,
            "hinge_status": bool_info.clone(),
            "agent_hinge_status": bool_info.clone(),
            "hinge_steps": float_info.clone(),
            "s": float_info.clone(),
            "vel": vel,
            "distance_ref": distance_ref,
            "rot": float_info.clone(),
            "act_acc": float_info.clone(),
            "steering_rate_abs_deg": float_info.clone(),
            "done_collision_with_agents": bool_info.clone(),
            "done_collision_with_lanelets": bool_info.clone(),
            "done_collision_with_exit_segments": bool_info.clone(),
        },
    }


def test_compute_validation_metrics_uses_all_frame_samples_for_error_stats():
    module = _load_occt_metrics_module()
    result_data = {
        "method": "unit_test",
        "followers": [1, 2],
        "dt": 1.0,
        "requested_road_id": 4,
        "road_name": "s_curve",
        "episodes_requested": 2,
        "episodes_completed": 2,
        "episodes": [
            _make_episode(front_errors=[0.0, 0.0], lateral_errors=[0.0, 0.0]),
            _make_episode(
                front_errors=[4.0, 4.0, 4.0, 4.0],
                lateral_errors=[2.0, 2.0, 2.0, 2.0],
            ),
        ],
    }

    metrics = module.compute_validation_metrics_from_object(
        result_data,
        followers=[1, 2],
        dt=1.0,
    )

    expected_lateral = torch.tensor([0.0, 0.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float32)
    expected_front = torch.tensor([0.0, 0.0, 4.0, 4.0, 4.0, 4.0], dtype=torch.float32)

    assert math.isclose(
        metrics["la_error_mean"],
        float(expected_lateral.mean().item()),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        metrics["la_error_std"],
        float(expected_lateral.std(unbiased=False).item()),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        metrics["s_error_mean"],
        float(expected_front.mean().item()),
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        metrics["s_error_std"],
        float(expected_front.std(unbiased=False).item()),
        rel_tol=0.0,
        abs_tol=1e-6,
    )

    # The old behavior computed std over per-episode means ([0, 2] and [0, 4]).
    assert not math.isclose(metrics["la_error_std"], 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert not math.isclose(metrics["s_error_std"], 2.0, rel_tol=0.0, abs_tol=1e-6)
