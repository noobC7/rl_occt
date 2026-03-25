# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from tqdm import tqdm
import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.record.loggers import generate_exp_name, get_logger, Logger
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.record.loggers.swanlab import SwanLabLogger


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute the mean of value over masked entries.

    Returns 0.0 when the mask selects no entries so logging remains stable.
    """
    if mask.ndim < value.ndim:
        mask = mask.reshape(*mask.shape, *([1] * (value.ndim - mask.ndim)))
    mask = mask.expand_as(value)
    selected = value[mask]
    if selected.numel() == 0:
        return 0.0
    return selected.float().mean().item()


def init_logging(cfg, model_name: str):
    experiment_name=generate_exp_name(model_name, cfg.logger.group_name)
    print(f"experiment_name: {experiment_name}")
    resume = False if cfg.logger.resume_swanlab_id=="None" else True
    logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name=os.getcwd(),
        experiment_name=experiment_name,
        wandb_kwargs={
            "group": cfg.logger.group_name or model_name,
            "project": cfg.logger.project_name
            or f"torchrl_example_{cfg.env.scenario_name}",
        },
        swanlab_kwargs={
            "config": cfg,
            "group": cfg.logger.group_name or model_name,
            "project": cfg.logger.project_name
            or f"swanlab_{cfg.env.scenario_name}",
            "mode": cfg.logger.mode,
            "id": cfg.logger.resume_swanlab_id if resume else None,
            "resume": resume,
        },
    )
    logger.log_hparams(cfg)
    return logger


def log_training(
    logger: Logger,
    training_td: TensorDictBase,
    sampling_td: TensorDictBase,
    sampling_time: float,
    training_time: float,
    total_time: float,
    iteration: int,
    current_frames: int,
    total_frames: int,
    step: int,
    extra_metrics: dict[str, float] | None = None,
):
    if ("next", "agents", "reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "reward"),
            sampling_td.get(("next", "reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )
    if ("next", "agents", "episode_reward") not in sampling_td.keys(True, True):
        sampling_td.set(
            ("next", "agents", "episode_reward"),
            sampling_td.get(("next", "episode_reward"))
            .expand(sampling_td.get("agents").shape)
            .unsqueeze(-1),
        )
    info_ignore_keys={"episode_done",
                      "done_all_hinged",
                      "done_collision_with_agents",
                      "done_collision_with_lanelets",
                      "done_collision_with_exit_segments",}
    metrics_to_log = {
        f"train/learner/{key}": value.mean().item()
        for key, value in training_td.items()
    }

    if "info" in sampling_td.get("agents").keys():
        info_td = sampling_td.get(("agents", "info"))
        next_info_td = sampling_td.get(("next", "agents", "info"), None)
        done_info_td = next_info_td if next_info_td is not None else info_td
        for key, value in info_td.items():
            if key not in info_ignore_keys:
                metrics_to_log.update(
                    {
                        f"train/info/{key}": value.mean().item()
                    }
                )
        hinge_status = info_td.get("hinge_status", None)
        if hinge_status is not None:
            hinge_mask = hinge_status.to(torch.bool)
            non_hinge_mask = ~hinge_mask
            hinge_count = hinge_mask.float().sum().item()
            total_phase_count = float(hinge_mask.numel())
            platoon_count = total_phase_count - hinge_count
            metrics_to_log["train/info/hinge_status_true_ratio"] = (
                hinge_mask.float().mean().item()
            )
            metrics_to_log["train/info/collector_hinge_sample_count"] = hinge_count
            metrics_to_log["train/info/collector_platoon_sample_count"] = platoon_count
            metrics_to_log["train/info/collector_hinge_sample_ratio"] = (
                hinge_count / max(total_phase_count, 1.0)
            )
            metrics_to_log["train/info/collector_platoon_sample_ratio"] = (
                platoon_count / max(total_phase_count, 1.0)
            )
            for key, value in info_td.items():
                if not isinstance(value, torch.Tensor) or not key.startswith("reward_"):
                    continue
                if "platoon" in key:
                    metrics_to_log[f"train/info/{key}"] = _masked_mean(
                        value, non_hinge_mask
                    )
                elif "hinge" in key:
                    metrics_to_log[f"train/info/{key}"] = _masked_mean(
                        value, hinge_mask
                    )
        episode_success = done_info_td.get("episode_success", None)
        episode_failure = done_info_td.get("episode_failure", None)
        actual_done = sampling_td.get(("next", "done"), None)
        if actual_done is not None:
            actual_done_mask = actual_done.to(torch.bool)
            actual_done_count = actual_done.float().sum().item()
            if episode_success is not None:
                success_rate = _masked_mean(
                    episode_success.float(), actual_done_mask
                )
                metrics_to_log["train/info/success_rate"] = success_rate
                metrics_to_log["train/info/episode_success_count"] = (
                    success_rate * actual_done_count
                )
            if episode_failure is not None:
                failure_rate = _masked_mean(
                    episode_failure.float(), actual_done_mask
                )
                metrics_to_log["train/info/failure_rate"] = failure_rate
                metrics_to_log["train/info/episode_failure_count"] = (
                    failure_rate * actual_done_count
                )
            for key in (
                "done_all_hinged",
                "done_collision_with_agents",
                "done_collision_with_lanelets",
                "done_collision_with_exit_segments",
            ):
                value = done_info_td.get(key, None)
                if value is not None:
                    metrics_to_log[f"train/info/{key}_rate"] = _masked_mean(
                        value.float(), actual_done_mask
                    )

    reward = sampling_td.get(("next", "agents", "reward")).mean(-2)  # Mean over agents
    done = sampling_td.get(("next", "done"))
    if done.ndim > reward.ndim:
        done = done[..., 0, :]  # Remove expanded agent dim
    episode_reward = sampling_td.get(("next", "agents", "episode_reward")).mean(-2)[
        done
    ]
    metrics_to_log.update(
        {
            "train/reward/reward_min": reward.min().item(),
            "train/reward/reward_mean": reward.mean().item(),
            "train/reward/reward_max": reward.max().item(),
            "train/reward/episode_reward_min": episode_reward.min().item(),
            "train/reward/episode_reward_mean": episode_reward.mean().item(),
            "train/reward/episode_reward_max": episode_reward.max().item(),
            "train/sampling_time": sampling_time,
            "train/training_time": training_time,
            "train/iteration_time": training_time + sampling_time,
            "train/total_time": total_time,
            "train/training_iteration": iteration,
            "train/current_frames": current_frames,
            "train/total_frames": total_frames,
        }
    )
    if extra_metrics:
        metrics_to_log.update(extra_metrics)
    try:
        env_total_step = sampling_td.get(("agents", "info", "env_total_step"))[:, :, 0, 0] #shape[batch_size, T]
        road_batch_id = sampling_td.get(("agents", "info", "road_batch_id"))[:, -1, 0, 0].to(torch.int64) #shape[batch_size]
        non_zero_mask = (env_total_step != 0).float()
        env_max_step = torch.max(env_total_step * non_zero_mask, dim=1)[0]
        # 初始化结果张量
        num_roads = torch.unique(road_batch_id).numel()
        road_mean_step = torch.zeros(num_roads, dtype=env_total_step.dtype, device=env_total_step.device)
        road_max_step = torch.zeros(num_roads, dtype=env_total_step.dtype, device=env_total_step.device)
        road_min_step = torch.full((num_roads,), float('inf'), dtype=env_total_step.dtype, device=env_total_step.device)
        # 均值
        sum_per_env = torch.sum(env_total_step * non_zero_mask, dim=1)
        count_per_env = torch.clamp(torch.sum(non_zero_mask, dim=1), min=1)
        sum_per_road = torch.zeros(num_roads, dtype=env_total_step.dtype, device=env_total_step.device)
        count_per_road = torch.zeros(num_roads, dtype=env_total_step.dtype, device=env_total_step.device)
        sum_per_road.scatter_reduce_(dim=0, index=road_batch_id, src=sum_per_env, reduce='sum', include_self=False)
        count_per_road.scatter_reduce_(dim=0, index=road_batch_id, src=count_per_env, reduce='sum', include_self=False)
        # 最大值
        road_mean_step = sum_per_road / count_per_road
        max_per_env = torch.max(env_total_step * non_zero_mask, dim=1)[0]
        road_max_step.scatter_reduce_(dim=0, index=road_batch_id, src=max_per_env, reduce='max', include_self=False)
        # 最小值
        env_non_zero_only = torch.where(non_zero_mask.bool(), env_total_step, torch.full_like(env_total_step, float('inf')))
        min_per_env = torch.min(env_non_zero_only, dim=1)[0]
        road_min_step.scatter_reduce_(dim=0, index=road_batch_id, src=min_per_env, reduce='min', include_self=False)
        road_min_step = torch.where(torch.isinf(road_min_step), torch.tensor(0.0, device=road_min_step.device), road_min_step)

        env_max_step = torch.max(max_per_env, dim=0)[0]
        env_min_step = torch.min(min_per_env, dim=0)[0]
        env_mean_step = torch.sum(sum_per_env, dim=0)/torch.sum(count_per_env, dim=0)
        metrics_to_log.update({
            "train/road_mean_step": road_mean_step.mean().float().item(),
            "train/road_max_step": road_max_step.mean().float().item(),
            "train/road_min_step": road_min_step.mean().float().item(),
            "train/env_max_step": env_max_step.float().item(),
            "train/env_min_step": env_min_step.float().item(),
            "train/env_mean_step": env_mean_step.float().item(),
        })
        for i in range(road_mean_step.shape[0]):
            metrics_to_log.update({
                f"train/road_{i}_mean_step": road_mean_step.float()[i].item(),
            })
            metrics_to_log.update({
                f"train/road_{i}_max_step": road_max_step.float()[i].item(),
            })
            metrics_to_log.update({
                f"train/road_{i}_min_step": road_min_step.float()[i].item(),
            })
        road_min_step_formatted = [int(round(x, 0)) for x in road_min_step.tolist()]
        road_mean_step_formatted = [int(round(x, 0)) for x in road_mean_step.tolist()]
        road_max_step_formatted = [int(round(x, 0)) for x in road_max_step.tolist()]
        print(f"Training - Total road min steps: {road_min_step_formatted}, "
            f"mean steps: {road_mean_step_formatted}, "
            f"max steps: {road_max_step_formatted}")
    except (KeyError, AttributeError) as e:
        # 如果 info 中没有 max_episode_step_reached 字段，跳过这个统计
        print(f"Warning: Could not extract max_episode_step_reached from info: {e}")
        pass

    if isinstance(logger, WandbLogger):
        logger.experiment.log(metrics_to_log, commit=False)
    else:
        for key, value in metrics_to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)

    return metrics_to_log


def log_evaluation(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    evaluation_time: float,
    step: int,
    video_caption: str = None,
):
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        next_done = next_done.clone()  # 必须clone，否则会修改原rollouts中的数据
        next_done[-1] = True  # 将最后一个位置设为True
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]
    rewards = [td.get(("next", "agents", "reward")).sum(0).mean() for td in rollouts]
    metrics_to_log = {
        "eval/episode_reward_min": min(rewards),
        "eval/episode_reward_max": max(rewards),
        "eval/episode_reward_mean": sum(rewards) / len(rollouts),
        "eval/episode_step_mean": sum([td.batch_size[0] for td in rollouts])
        / len(rollouts),
        "eval/evaluation_time": evaluation_time,
    }
    road_total_step=[]
    for i,td in enumerate(rollouts):
        metrics_to_log.update({
            f"eval/road_{i}_total_step": td.batch_size[0],
        })  
        road_total_step.append(td.batch_size[0])
    print(f"Evaluation - Total road steps per env: {road_total_step}")
    vid = torch.tensor(
        np.transpose(env_test.frames[: rollouts[0].batch_size[0]], (0, 3, 1, 2)),
        dtype=torch.uint8,
    ).unsqueeze(0)

    if isinstance(logger, WandbLogger):
        import wandb

        logger.experiment.log(metrics_to_log, commit=False)
        logger.experiment.log(
            {
                "eval/video": wandb.Video(vid, fps=1 / env_test.world.dt, format="mp4"),
            },
            commit=False,
        )
    if isinstance(logger, SwanLabLogger):
        logger.experiment.log(metrics_to_log)
        logger.log_video("eval/video", vid, step=step, caption=video_caption, fps=int(1/env_test.world.dt))
    else:
        for key, value in metrics_to_log.items():
            logger.log_scalar(key.replace("/", "_"), value, step=step)
        logger.log_video("eval_video", vid, step=step, fps=int(1/env_test.world.dt))


def log_batch_video(
    logger: WandbLogger,
    rollouts: TensorDictBase,
    env_test: VmasEnv,
    iter: int = None,
):
    rollouts = list(rollouts.unbind(0))
    for k, r in enumerate(rollouts):
        next_done = r.get(("next", "done")).sum(
            tuple(range(r.batch_dims, r.get(("next", "done")).ndim)),
            dtype=torch.bool,
        )
        next_done = next_done.clone()  # 必须clone，否则会修改原rollouts中的数据
        next_done[-1] = True  # 将最后一个位置设为True
        done_index = next_done.nonzero(as_tuple=True)[0][
            0
        ]  # First done index for this traj
        rollouts[k] = r[: done_index + 1]
    assert isinstance(logger, SwanLabLogger), "logger must be SwanLabLogger"
    for env_index in tqdm(range(env_test.num_envs), desc="Encoding videos"):
        road_id = env_index
        if hasattr(env_test.scenario, "road") and hasattr(env_test.scenario.road, "batch_id"):
            road_id = int(env_test.scenario.road.batch_id[env_index].item())
        frame_array = np.stack(
            env_test.frames[env_index][: rollouts[env_index].batch_size[0]], axis=0
        )
        vid = torch.tensor(
            frame_array.transpose(0, 3, 1, 2),
            dtype=torch.uint8,
        ).unsqueeze(0)
        logger.log_mp4_local(vid, caption=f"iter_{iter}_path_{road_id}.mp4", fps = int(1/env_test.world.dt))
