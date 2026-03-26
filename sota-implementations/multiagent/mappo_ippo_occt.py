import time
import os
import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.modules.models.multiagent import (
    MultiAgentMLP,
    PhaseConditionedMultiAgentMLP,
)
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from utils.logging import init_logging, log_evaluation, log_training, log_batch_video
from utils.utils import DoneTransform, save_checkpoint, save_rollout, load_checkpoint
AGENT_FOCUS_INDEX=2
def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=AGENT_FOCUS_INDEX))

def rendering_batch_callback(env, td):
    for env_index in range(env.num_envs):
        #env.frames[env_index].append(env.render(mode="rgb_array", agent_index_focus=round(env.scenario.n_agents/2)-1, env_index=env_index)) 
        env.frames[env_index].append(env.render(mode="rgb_array", agent_index_focus=AGENT_FOCUS_INDEX, env_index=env_index)) 


def build_eval_env(cfg_test: DictConfig, num_envs: int) -> VmasEnv:
    return VmasEnv(
        scenario=cfg_test.env.scenario_name,
        num_envs=num_envs,
        continuous_actions=True,
        max_steps=cfg_test.env.eval_max_steps,
        device=cfg_test.env.device,
        seed=cfg_test.seed,
        **cfg_test.env.scenario,
    )


def _safe_get(td, key):
    try:
        return td.get(key)
    except KeyError:
        return None


def _ensure_phase_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype is not torch.bool:
        mask = mask > 0.5
    if mask.ndim == 0:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != 1:
        mask = mask.unsqueeze(-1)
    return mask


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while mask.ndim > target.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim < target.ndim:
        mask = mask.reshape(*mask.shape, *([1] * (target.ndim - mask.ndim)))
    if mask.ndim != target.ndim:
        raise RuntimeError(
            f"Mask/target ndim mismatch after alignment: mask={tuple(mask.shape)}, "
            f"target={tuple(target.shape)}"
        )
    return mask.expand_as(target)


def _masked_tensor_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        return torch.as_tensor(value, device=mask.device)
    if value.ndim == 0:
        return value
    expanded_mask = _expand_mask(mask, value)
    if bool(expanded_mask.any().item()):
        return value[expanded_mask].mean()
    return value.sum() * 0.0


def _reduce_metric(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.mean()
    return torch.as_tensor(value)


def _safe_info_get(td, info_key: str):
    value = _safe_get(td, ("agents", "info", info_key))
    if value is None:
        value = _safe_get(td, ("next", "agents", "info", info_key))
    return value


def infer_observation_group_slices(
    vmas_env: VmasEnv, agent_index: int = 0
) -> dict[str, slice]:
    agent_index = min(max(agent_index, 0), vmas_env.n_agents - 1)
    td = vmas_env.reset()
    scenario = vmas_env.scenario
    _, obs_self_groups = scenario.observe_self(agent_index, return_groups=True)
    _, obs_other_groups = scenario.observe_other_agents_platoon(
        agent_index, return_groups=True
    )
    total_obs_dim = td["agents", "observation"].shape[-1]

    group_slices = {}
    cursor = 0
    for name, tensor in [*obs_self_groups, *obs_other_groups]:
        if tensor is None:
            continue
        next_cursor = cursor + int(tensor.shape[-1])
        group_slices[name] = slice(cursor, next_cursor)
        cursor = next_cursor

    if cursor > total_obs_dim:
        raise RuntimeError(
            f"Observation groups exceed total observation dim: {cursor} > {total_obs_dim}."
        )
    return group_slices


def infer_agent_advantage_exclude_dims(
    td_batch_size: torch.Size,
    advantage_shape: torch.Size,
    n_agents: int,
) -> tuple[int, ...]:
    batch_ndim = len(td_batch_size)
    extra_dims = list(range(batch_ndim, len(advantage_shape)))
    if not extra_dims:
        return ()

    agent_dims = [dim for dim in extra_dims if advantage_shape[dim] == n_agents]
    if len(agent_dims) == 1:
        agent_dim = agent_dims[0]
    else:
        non_singleton_extra_dims = [dim for dim in extra_dims if advantage_shape[dim] > 1]
        if len(non_singleton_extra_dims) != 1:
            return ()
        agent_dim = non_singleton_extra_dims[0]

    return (agent_dim - len(advantage_shape),)


def log_advantage_layout(tag: str, td, loss_module, n_agents: int) -> None:
    advantage_key = loss_module.tensor_keys.advantage
    advantage = _safe_get(td, advantage_key)
    observation = _safe_get(td, ("agents", "observation"))
    reward = _safe_get(td, ("next", "agents", "reward"))
    hinge_status = _safe_get(td, ("agents", "info", "hinge_status"))
    print(
        f"[PPO][{tag}] td.batch_size={tuple(td.batch_size)}, "
        f"adv_key={advantage_key}, "
        f"adv_shape={None if advantage is None else tuple(advantage.shape)}, "
        f"obs_shape={None if observation is None else tuple(observation.shape)}, "
        f"reward_shape={None if reward is None else tuple(reward.shape)}, "
        f"hinge_status_shape={None if hinge_status is None else tuple(hinge_status.shape)}, "
        f"n_agents={n_agents}, "
        f"normalize_advantage_exclude_dims={loss_module.normalize_advantage_exclude_dims}"
    )


def extract_training_phase_masks(
    td,
    hinge_key: str = "hinge_status",
    agent_hinged_key: str = "agent_hinge_status",
) -> tuple[torch.Tensor, torch.Tensor]:
    hinge_status = _safe_info_get(td, hinge_key)
    agent_hinged = _safe_info_get(td, agent_hinged_key)
    if hinge_status is None or agent_hinged is None:
        raise KeyError(
            "Could not find hinge_status/agent_hinge_status in env info. "
            "Phase-conditioned training requires these signals."
        )
    hinge_status = _ensure_phase_mask(hinge_status)
    agent_hinged = _ensure_phase_mask(agent_hinged)
    hinge_approach_mask = hinge_status & ~agent_hinged
    platoon_mask = (~hinge_status) & ~agent_hinged
    return platoon_mask, hinge_approach_mask


class MetricAdaptiveWeightController:
    def __init__(self, cfg: DictConfig, device: torch.device | str) -> None:
        adaptive_cfg = cfg.get("adaptive_weighting", {})
        self.enabled = bool(adaptive_cfg.get("enabled", False))
        self.ema_tau = float(adaptive_cfg.get("ema_tau", 0.1))
        self.initial_hinge_weight = float(
            adaptive_cfg.get("initial_hinge_weight", 0.0)
        )
        self.min_platoon_weight = float(
            adaptive_cfg.get("min_platoon_weight", 0.2)
        )
        self.max_hinge_weight = float(
            adaptive_cfg.get("max_hinge_weight", 0.8)
        )
        self.collision_high_threshold = float(
            adaptive_cfg.get("collision_high_threshold", 0.20)
        )
        self.collision_low_threshold = float(
            adaptive_cfg.get("collision_low_threshold", 0.08)
        )
        self.platoon_space_error_threshold = float(
            adaptive_cfg.get("platoon_space_error_threshold", 0.60)
        )
        self.platoon_ref_error_threshold = float(
            adaptive_cfg.get("platoon_ref_error_threshold", 0.20)
        )
        self.all_hinged_target = float(adaptive_cfg.get("all_hinged_target", 0.85))
        self.all_hinged_plateau_delta = float(
            adaptive_cfg.get("all_hinged_plateau_delta", 0.01)
        )
        self.plateau_patience = int(adaptive_cfg.get("plateau_patience", 3))
        self.hinge_weight_step = float(adaptive_cfg.get("hinge_weight_step", 0.05))
        self.collision_weight_step = float(
            adaptive_cfg.get("collision_weight_step", 0.10)
        )
        self.device = torch.device(device)

        self.hinge_weight = self.initial_hinge_weight
        self.collision_rate_ema = None
        self.platoon_space_error_ema = None
        self.platoon_ref_error_ema = None
        self.all_hinged_rate_ema = None
        self.plateau_counter = 0

    def _update_ema(self, current: torch.Tensor, attr_name: str) -> torch.Tensor:
        previous = getattr(self, attr_name)
        if previous is None:
            updated = current.detach()
        else:
            updated = torch.lerp(previous, current.detach(), self.ema_tau)
        setattr(self, attr_name, updated)
        return updated

    def _make_weight_tensor(
        self, platoon_weight: float, hinge_weight: float, device: torch.device
    ) -> dict[str, torch.Tensor]:
        weight_sum = max(platoon_weight + hinge_weight, 1e-8)
        return {
            "platoon": torch.tensor(
                platoon_weight / weight_sum, device=device, dtype=torch.float32
            ),
            "hinge_approach": torch.tensor(
                hinge_weight / weight_sum, device=device, dtype=torch.float32
            ),
        }

    def collector_state(self, td) -> dict[str, object]:
        platoon_mask, hinge_approach_mask = extract_training_phase_masks(td)
        done_mask = _safe_info_get(td, "episode_done")
        done_all_hinged = _safe_info_get(td, "done_all_hinged")
        collision_agents = _safe_info_get(td, "done_collision_with_agents")
        collision_lanelets = _safe_info_get(td, "done_collision_with_lanelets")
        collision_exit = _safe_info_get(td, "done_collision_with_exit_segments")
        error_space = _safe_info_get(td, "error_space")
        distance_ref = _safe_info_get(td, "distance_ref")

        done_mask = _ensure_phase_mask(done_mask) if done_mask is not None else None
        done_all_hinged = (
            _ensure_phase_mask(done_all_hinged) if done_all_hinged is not None else None
        )
        collision_any = None
        if (
            collision_agents is not None
            and collision_lanelets is not None
            and collision_exit is not None
        ):
            collision_any = (
                _ensure_phase_mask(collision_agents)
                | _ensure_phase_mask(collision_lanelets)
                | _ensure_phase_mask(collision_exit)
            )

        if done_mask is not None and bool(done_mask.any().item()):
            collision_rate = (
                collision_any[done_mask].float().mean()
                if collision_any is not None
                else torch.zeros((), device=self.device)
            )
            all_hinged_rate = (
                done_all_hinged[done_mask].float().mean()
                if done_all_hinged is not None
                else torch.zeros((), device=self.device)
            )
        else:
            collision_rate = (
                collision_any.float().mean()
                if collision_any is not None
                else torch.zeros((), device=self.device)
            )
            all_hinged_rate = torch.zeros((), device=self.device)

        if error_space is not None:
            platoon_space_error = _masked_tensor_mean(
                torch.abs(error_space[..., 0]),
                platoon_mask,
            )
        else:
            platoon_space_error = torch.zeros((), device=self.device)

        if distance_ref is not None:
            platoon_ref_error = _masked_tensor_mean(
                torch.abs(distance_ref),
                platoon_mask,
            )
        else:
            platoon_ref_error = torch.zeros((), device=self.device)

        collision_rate_ema = self._update_ema(collision_rate, "collision_rate_ema")
        platoon_space_error_ema = self._update_ema(
            platoon_space_error, "platoon_space_error_ema"
        )
        platoon_ref_error_ema = self._update_ema(
            platoon_ref_error, "platoon_ref_error_ema"
        )
        previous_all_hinged_ema = self.all_hinged_rate_ema
        all_hinged_rate_ema = self._update_ema(
            all_hinged_rate, "all_hinged_rate_ema"
        )
        all_hinged_improvement = (
            0.0
            if previous_all_hinged_ema is None
            else float((all_hinged_rate_ema - previous_all_hinged_ema).item())
        )

        stable_platoon = bool(
            collision_rate_ema.item() < self.collision_low_threshold
            and platoon_space_error_ema.item() < self.platoon_space_error_threshold
            and platoon_ref_error_ema.item() < self.platoon_ref_error_threshold
        )

        if not self.enabled:
            weights = self._make_weight_tensor(1.0, 0.0, platoon_mask.device)
        else:
            if collision_rate_ema.item() > self.collision_high_threshold:
                self.hinge_weight = max(
                    0.0, self.hinge_weight - self.collision_weight_step
                )
                self.plateau_counter = 0
            elif (
                stable_platoon
                and all_hinged_rate_ema.item() < self.all_hinged_target
                and all_hinged_improvement < self.all_hinged_plateau_delta
            ):
                self.plateau_counter += 1
                if self.plateau_counter >= self.plateau_patience:
                    self.hinge_weight = min(
                        self.max_hinge_weight,
                        self.hinge_weight + self.hinge_weight_step,
                    )
                    self.plateau_counter = 0
            else:
                self.plateau_counter = 0

            platoon_weight = max(self.min_platoon_weight, 1.0 - self.hinge_weight)
            hinge_weight = min(self.max_hinge_weight, 1.0 - platoon_weight)
            weights = self._make_weight_tensor(
                platoon_weight,
                hinge_weight,
                platoon_mask.device,
            )

        metrics = {
            "train/adaptive_weighting/enabled": float(self.enabled),
            "train/adaptive_weighting/w_platoon": float(weights["platoon"].item()),
            "train/adaptive_weighting/w_hinge_approach": float(
                weights["hinge_approach"].item()
            ),
            "train/adaptive_weighting/collision_rate": float(collision_rate.item()),
            "train/adaptive_weighting/collision_rate_ema": float(
                collision_rate_ema.item()
            ),
            "train/adaptive_weighting/all_hinged_rate": float(all_hinged_rate.item()),
            "train/adaptive_weighting/all_hinged_rate_ema": float(
                all_hinged_rate_ema.item()
            ),
            "train/adaptive_weighting/all_hinged_improvement": all_hinged_improvement,
            "train/adaptive_weighting/platoon_space_error": float(
                platoon_space_error.item()
            ),
            "train/adaptive_weighting/platoon_space_error_ema": float(
                platoon_space_error_ema.item()
            ),
            "train/adaptive_weighting/platoon_ref_error": float(
                platoon_ref_error.item()
            ),
            "train/adaptive_weighting/platoon_ref_error_ema": float(
                platoon_ref_error_ema.item()
            ),
            "train/adaptive_weighting/platoon_stable": float(stable_platoon),
            "train/adaptive_weighting/plateau_counter": float(self.plateau_counter),
            "train/adaptive_weighting/platoon_ratio": float(
                platoon_mask.float().mean().item()
            ),
            "train/adaptive_weighting/hinge_approach_ratio": float(
                hinge_approach_mask.float().mean().item()
            ),
        }
        return {"weights": weights, "metrics": metrics}


def build_training_summary(
    loss_vals,
    platoon_mask: torch.Tensor,
    hinge_approach_mask: torch.Tensor,
    phase_weights: dict[str, torch.Tensor] | None,
) -> tuple[torch.Tensor, TensorDict]:
    summary = TensorDict({}, batch_size=[])

    if phase_weights is None:
        component_totals = []
        for key in ("loss_objective", "loss_critic", "loss_entropy"):
            if key not in loss_vals.keys():
                continue
            component_value = _reduce_metric(loss_vals[key])
            summary.set(key, component_value.detach())
            component_totals.append(component_value)
        loss_value = sum(component_totals)
        summary.set(
            "adaptive_weight_platoon", torch.tensor(1.0, device=loss_value.device)
        )
        summary.set(
            "adaptive_weight_hinge_approach",
            torch.tensor(0.0, device=loss_value.device),
        )
    else:
        weighted_component_totals = {}
        platoon_has_samples = bool(platoon_mask.any().item())
        hinge_has_samples = bool(hinge_approach_mask.any().item())
        if platoon_has_samples and hinge_has_samples:
            platoon_weight = phase_weights["platoon"]
            hinge_weight = phase_weights["hinge_approach"]
        elif platoon_has_samples:
            platoon_weight = torch.tensor(1.0, device=platoon_mask.device)
            hinge_weight = torch.tensor(0.0, device=platoon_mask.device)
        elif hinge_has_samples:
            platoon_weight = torch.tensor(0.0, device=platoon_mask.device)
            hinge_weight = torch.tensor(1.0, device=platoon_mask.device)
        else:
            platoon_weight = torch.tensor(1.0, device=platoon_mask.device)
            hinge_weight = torch.tensor(0.0, device=platoon_mask.device)
        for key in ("loss_objective", "loss_critic", "loss_entropy"):
            if key not in loss_vals.keys():
                continue
            platoon_component = _masked_tensor_mean(loss_vals[key], platoon_mask)
            hinge_component = _masked_tensor_mean(
                loss_vals[key], hinge_approach_mask
            )
            weighted_component = (
                platoon_weight * platoon_component
                + hinge_weight * hinge_component
            )
            weighted_component_totals[key] = weighted_component
            summary.set(key, weighted_component.detach())
            summary.set(f"{key}_platoon", platoon_component.detach())
            summary.set(f"{key}_hinge_approach", hinge_component.detach())
        loss_value = sum(weighted_component_totals.values())
        summary.set("adaptive_weight_platoon", platoon_weight.detach())
        summary.set(
            "adaptive_weight_hinge_approach",
            hinge_weight.detach(),
        )

    summary.set("adaptive_ratio_platoon", platoon_mask.float().mean().detach())
    summary.set(
        "adaptive_ratio_hinge_approach",
        hinge_approach_mask.float().mean().detach(),
    )
    for metric_key in (
        "entropy",
        "clip_fraction",
        "kl_approx",
        "ESS",
        "explained_variance",
    ):
        metric_value = loss_vals.get(metric_key, None)
        if metric_value is not None:
            summary.set(metric_key, _reduce_metric(metric_value).detach())

    return loss_value, summary


def configure_eval_env_paths(env_test: VmasEnv, path_ids: list[int]) -> None:
    road = env_test.scenario.road
    full_path_library = list(road.path_library)
    if not path_ids:
        raise ValueError("path_ids must not be empty for evaluation.")
    if max(path_ids) >= len(full_path_library):
        raise ValueError(
            f"Requested path id {max(path_ids)} exceeds available paths {len(full_path_library)}."
        )

    road.path_library = [full_path_library[path_id] for path_id in path_ids]
    road.reset_splines()
    road.batch_id = torch.tensor(path_ids, dtype=torch.int64, device=road.device)

    scenario = env_test.scenario
    scenario.road_total_step = scenario.env_total_step.new_zeros(len(full_path_library))
    scenario.lane_width = road.get_lane_width("mean")
    scenario.ref_paths_agent_related.long_term = road.get_road_center_pts().unsqueeze(1).expand(
        -1, scenario.n_agents, -1, -1
    )
    scenario.ref_paths_agent_related.left_boundary = road.get_road_left_pts().unsqueeze(1).expand(
        -1, scenario.n_agents, -1, -1
    )
    scenario.ref_paths_agent_related.right_boundary = road.get_road_right_pts().unsqueeze(1).expand(
        -1, scenario.n_agents, -1, -1
    )
    env_test.reset()


def run_eval_export_chunk(
    logger,
    policy,
    env_test: VmasEnv,
    start_iteration: int,
    total_frames: int,
    max_steps: int,
    chunk_index: int,
    total_chunks: int,
):
    path_ids = env_test.scenario.road.batch_id.detach().cpu().tolist()
    print(
        f"[OcctCRMap] Eval chunk {chunk_index + 1}/{total_chunks}, paths: {path_ids}. Start rollout."
    )
    evaluation_start = time.time()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        env_test.frames = [[] for _ in range(env_test.num_envs)]
        rollouts = env_test.rollout(
            max_steps=max_steps,
            policy=policy,
            callback=rendering_batch_callback,
            auto_cast_to_device=True,
            break_when_any_done=False,
            break_when_all_done=True,
        )
        rollout_suffix = f"_paths_{path_ids[0]}_{path_ids[-1]}"
        save_rollout(logger, rollouts, start_iteration, total_frames, suffix=rollout_suffix)
        evaluation_time = time.time() - evaluation_start
        print(
            f"[OcctCRMap] Eval chunk {chunk_index + 1}/{total_chunks} rollout finished, "
            f"duration: {evaluation_time:.2f}s."
        )
        log_batch_video(logger, rollouts, env_test, iter=start_iteration)
        video_encode_time = time.time() - evaluation_start - evaluation_time
        print(
            f"[OcctCRMap] Eval chunk {chunk_index + 1}/{total_chunks} video encode finished, "
            f"duration: {video_encode_time:.2f}s."
        )

    env_test.frames = None
    del rollouts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@hydra.main(version_base="1.1", config_path="config", config_name="mappo_mlp_continues_act_change_penalty")
def train(cfg: DictConfig):
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.max_steps = eval(cfg.env.max_steps)
    print(cfg.env)
    torch.manual_seed(cfg.seed)
    resume_from_checkpoint = cfg.train.resume_from_checkpoint
    resume_mode = cfg.train.resume_mode
    start_iteration = 0
    start_frames = 0
    # Sampling
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    # Create env and env_test
    env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    cfg_test = cfg.copy()
    cfg_test.env.scenario.is_rand_arc_pos = False
    cfg_test.env.scenario.init_vel_std = 0
    env_test = None

    share_params = True if cfg.model.shared_parameters else False
    use_phase_conditioned_network = bool(
        cfg.model.get("use_phase_conditioned_network", False)
    )
    # Policy
    actor_depth = int(cfg.model.get("actor_depth", 2))
    actor_num_cells = cfg.model.get("actor_num_cells", 344)
    actor_head_hidden = cfg.model.get(
        "actor_head_hidden", cfg.model.get("actor_module_hidden", actor_num_cells)
    )
    critic_depth = int(cfg.model.get("critic_depth", 2))
    critic_num_cells = cfg.model.get("critic_num_cells", 688)
    critic_head_hidden = cfg.model.get(
        "critic_head_hidden", cfg.model.get("critic_module_hidden", critic_num_cells)
    )
    if use_phase_conditioned_network and not bool(cfg.model.get("centralised_critic", True)):
        torchrl_logger.warning(
            "Phase-conditioned OCCT training always uses a centralized critic. "
            "Overriding model.centralised_critic=False."
        )
    phase_slice = None
    if use_phase_conditioned_network:
        observation_env = env.base_env if hasattr(env, "base_env") else env
        observation_group_slices = infer_observation_group_slices(
            observation_env, agent_index=min(AGENT_FOCUS_INDEX, env.n_agents - 1)
        )
        phase_obs_group = cfg.model.get("phase_obs_group", "self_hinge_status")
        if phase_obs_group not in observation_group_slices:
            raise KeyError(
                f"Observation group '{phase_obs_group}' not found. Available groups: "
                f"{sorted(observation_group_slices.keys())}"
            )
        phase_slice = observation_group_slices[phase_obs_group]

    if use_phase_conditioned_network:
        actor_backbone = PhaseConditionedMultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2
            * env.full_action_spec_unbatched[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_params,
            device=cfg.train.device,
            depth=actor_depth,
            num_cells=actor_num_cells,
            head_hidden=actor_head_hidden,
            activation_class=nn.Tanh,
            phase_slice=phase_slice,
        )
    else:
        actor_backbone = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2
            * env.full_action_spec_unbatched[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_params,
            device=cfg.train.device,
            depth=actor_depth,
            num_cells=actor_num_cells,
            activation_class=nn.Tanh,
        )
    actor_net = nn.Sequential(actor_backbone, NormalParamExtractor())
    policy_module = TensorDictModule(
        actor_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.full_action_spec_unbatched,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[("agents", "action")].space.low,
            "high": env.full_action_spec_unbatched[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )

    # Critic
    if use_phase_conditioned_network:
        module = PhaseConditionedMultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralised=True,
            share_params=share_params,
            device=cfg.train.device,
            depth=critic_depth,
            num_cells=critic_num_cells,
            head_hidden=critic_head_hidden,
            activation_class=nn.Tanh,
            phase_slice=phase_slice,
        )
    else:
        module = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralised=cfg.model.centralised_critic,
            share_params=share_params,
            device=cfg.train.device,
            depth=critic_depth,
            num_cells=critic_num_cells,
            activation_class=nn.Tanh,
        )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation")],
    )
    phase_weight_controller = MetricAdaptiveWeightController(cfg, cfg.train.device)
    if cfg.collector.n_iters > 0:
        collector = SyncDataCollector(
            env,
            policy,
            device=cfg.env.device,
            storing_device=cfg.train.device,
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
        )

        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(cfg.buffer.memory_size, device=cfg.train.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=cfg.train.minibatch_size,
        )

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_eps,
        normalize_advantage=True,
        normalize_advantage_exclude_dims=(-2,),
        reduction="none",
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=cfg.loss.gamma, lmbda=cfg.loss.lmbda
    )
    optim = torch.optim.Adam(loss_module.parameters(), cfg.train.lr)

    if os.path.exists(resume_from_checkpoint):
        if resume_mode == "resume":
            start_iteration, start_frames = load_checkpoint(
                resume_from_checkpoint, policy, value_module, optim
            )
        elif resume_mode == "warm_start":
            _ = load_checkpoint(resume_from_checkpoint, policy, value_module, optim=None)
            start_iteration, start_frames = 0, 0
        elif resume_mode == "fine_tune":
            _ = load_checkpoint(resume_from_checkpoint, policy, value_module=None, optim=None)
            start_iteration, start_frames = 0, 0
        else:
            raise TypeError
        torchrl_logger.info(
            f"Resumed training from checkpoint {resume_from_checkpoint} "
            f"at iteration {start_iteration} and frame {start_frames}."
        )
    else:
        torchrl_logger.warning(
            f"Checkpoint path {resume_from_checkpoint} does not exist. "
            "Training from scratch."
        )
    # Logging
    if cfg.logger.backend:
        model_prefix = "PhaseMAPPO" if use_phase_conditioned_network else "MAPPO"
        if not use_phase_conditioned_network and not cfg.model.centralised_critic:
            model_prefix = "IPPO"
        model_name = ("Het" if not cfg.model.shared_parameters else "") + model_prefix
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = start_frames
    if cfg.collector.n_iters == 0:
        probe_env = build_eval_env(cfg_test, 1)
        total_path_num = probe_env.scenario.get_occt_cr_path_num()
        if not probe_env.is_closed:
            probe_env.close()

        total_eval_paths = min(cfg_test.eval.evaluation_episodes, total_path_num)
        render_batch_size = int(
            cfg_test.eval.get("render_batch_size", cfg_test.eval.evaluation_episodes)
        )
        render_batch_size = max(1, min(render_batch_size, total_eval_paths))
        total_chunks = (total_eval_paths + render_batch_size - 1) // render_batch_size

        print(
            f"[OcctCRMap] Path num: {total_path_num}. "
            f"Evaluating {total_eval_paths} paths with render_batch_size={render_batch_size}."
        )

        for chunk_index, chunk_start in enumerate(range(0, total_eval_paths, render_batch_size)):
            path_ids = list(
                range(chunk_start, min(chunk_start + render_batch_size, total_eval_paths))
            )
            env_test = build_eval_env(cfg_test, len(path_ids))
            configure_eval_env_paths(env_test, path_ids)
            try:
                run_eval_export_chunk(
                    logger=logger,
                    policy=policy,
                    env_test=env_test,
                    start_iteration=start_iteration,
                    total_frames=total_frames,
                    max_steps=cfg.env.eval_max_steps,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                )
            finally:
                if not env_test.is_closed:
                    env_test.close()

        if not env.is_closed:
            env.close()
        return

    env_test = build_eval_env(cfg_test, cfg_test.eval.evaluation_episodes)

    advantage_layout_logged = False
    minibatch_layout_logged = False
    sampling_start = time.time()
    pbar = tqdm(enumerate(collector, start=start_iteration), 
             initial=start_iteration,
             total=cfg.collector.n_iters, 
             desc="Training", 
             unit="iter")
    for i, tensordict_data in pbar:
        sampling_time = time.time() - sampling_start
        with torch.no_grad():
            loss_module.value_estimator(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        if not advantage_layout_logged:
            advantage = tensordict_data.get(loss_module.tensor_keys.advantage)
            loss_module.normalize_advantage_exclude_dims = infer_agent_advantage_exclude_dims(
                tensordict_data.batch_size,
                advantage.shape,
                env.n_agents,
            )
            log_advantage_layout("collector", tensordict_data, loss_module, env.n_agents)
            if not loss_module.normalize_advantage_exclude_dims:
                print(
                    "[PPO] Could not infer the agent dimension for advantage normalization. "
                    "Falling back to global advantage normalization."
                )
            advantage_layout_logged = True
        phase_state = phase_weight_controller.collector_state(tensordict_data)
        iteration_extra_metrics = phase_state["metrics"]
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for epoch_idx in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                pbar.set_postfix({"Epoch": f"{epoch_idx+1}/{cfg.train.num_epochs}"})
                subdata = replay_buffer.sample()
                if not minibatch_layout_logged:
                    log_advantage_layout("minibatch", subdata, loss_module, env.n_agents)
                    minibatch_layout_logged = True
                loss_vals = loss_module(subdata)
                platoon_mask, hinge_approach_mask = extract_training_phase_masks(
                    subdata
                )
                active_phase_weights = (
                    phase_state["weights"] if phase_weight_controller.enabled else None
                )
                loss_value, training_summary = build_training_summary(
                    loss_vals,
                    platoon_mask,
                    hinge_approach_mask,
                    active_phase_weights,
                )
                training_tds.append(training_summary.detach())

                loss_value.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.train.max_grad_norm
                )
                training_tds[-1].set("grad_norm", total_norm.mean())

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        training_time = time.time() - training_start

        iteration_time = sampling_time + training_time
        total_time += iteration_time
        training_tds = torch.stack(training_tds)

        # More logs
        if cfg.logger.backend:
            log_training(
                logger,
                training_tds,
                tensordict_data,
                sampling_time,
                training_time,
                total_time,
                i,
                current_frames,
                total_frames,
                step=i,
                extra_metrics=iteration_extra_metrics,
            )

        if (
            cfg.eval.evaluation_episodes > 0
            and (i % cfg.eval.evaluation_interval == 0 or i == cfg.collector.n_iters - 1)
            and cfg.logger.backend
        ):
            evaluation_start = time.time()
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                env_test.frames = []
                rollouts = env_test.rollout(
                    max_steps=cfg.env.eval_max_steps,
                    policy=policy,
                    callback=rendering_callback,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    # We are running vectorized evaluation we do not want it to stop when just one env is done
                )
                evaluation_time = time.time() - evaluation_start
                log_evaluation(logger, rollouts, env_test, evaluation_time, 
                               step=i, video_caption=f"path_0")
                save_checkpoint(logger, policy, value_module, optim, i, total_frames)
                save_rollout(logger, rollouts, i, total_frames)

        sampling_start = time.time()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()
if __name__ == "__main__":
    train()
