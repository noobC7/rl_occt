import time
import os
import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm
from tensordict import TensorDict, TensorDictBase
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


def _compute_total_grad_norm(
    parameters: list[torch.nn.Parameter], norm_type: float = 2.0
) -> torch.Tensor:
    if not parameters:
        return torch.zeros((), dtype=torch.float32)
    gradients = [param.grad.detach() for param in parameters if param.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float32, device=parameters[0].device)

    if norm_type == float("inf"):
        return torch.stack([grad.abs().max() for grad in gradients]).max()

    grad_norms = torch.stack([torch.norm(grad, norm_type) for grad in gradients])
    return torch.norm(grad_norms, norm_type)


def _reduce_metric(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.mean()
    return torch.as_tensor(value)


def _safe_info_get(td, info_key: str):
    value = _safe_get(td, ("agents", "info", info_key))
    if value is None:
        value = _safe_get(td, ("next", "agents", "info", info_key))
    return value


def _safe_observation_field_get(td, field_key: str):
    value = _safe_get(td, ("agents", "observation", field_key))
    if value is None:
        value = _safe_get(td, ("next", "agents", "observation", field_key))
    return value


def _select_latest_phase_frame(
    value: torch.Tensor | None, reference_ndim: int | None = None
) -> torch.Tensor | None:
    if value is None:
        return None
    if reference_ndim is not None and value.ndim > reference_ndim:
        return value.select(-2, 0)
    return value


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
    hinge_status = _safe_observation_field_get(td, "self_hinge_status")
    if hinge_status is None:
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
    hinge_key: str = "self_hinge_status",
    agent_hinged_key: str = "agent_hinge_status",
) -> tuple[torch.Tensor, torch.Tensor]:
    hinge_status = _safe_observation_field_get(td, hinge_key)
    if hinge_status is None and hinge_key == "self_hinge_status":
        hinge_status = _safe_info_get(td, "hinge_status")
    elif hinge_status is None:
        hinge_status = _safe_info_get(td, hinge_key)
    agent_hinged = _safe_info_get(td, agent_hinged_key)
    hinge_status = _select_latest_phase_frame(
        hinge_status,
        reference_ndim=None if agent_hinged is None else agent_hinged.ndim,
    )
    if hinge_status is None or agent_hinged is None:
        raise KeyError(
            "Could not find self_hinge_status(or fallback hinge_status)/agent_hinge_status. "
            "Phase-conditioned training requires these signals."
        )
    hinge_status = _ensure_phase_mask(hinge_status)
    agent_hinged = _ensure_phase_mask(agent_hinged)
    if hinge_status.shape != agent_hinged.shape:
        print(
            "[PHASE_MASK_DEBUG] "
            f"hinge_key={hinge_key}, "
            f"hinge_status_shape={tuple(hinge_status.shape)}, "
            f"agent_hinged_shape={tuple(agent_hinged.shape)}"
        )
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


def build_mlp(
    in_features: int,
    out_features: int,
    hidden_sizes: list[int],
    activation_class: type[nn.Module] = nn.Tanh,
    *,
    activate_last: bool = False,
) -> nn.Sequential:
    sizes = [in_features, *hidden_sizes, out_features]
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
        is_last_layer = idx == len(sizes) - 2
        if not is_last_layer or activate_last:
            layers.append(activation_class())
    return nn.Sequential(*layers)


def resolve_hidden_sizes(
    depth: int,
    num_cells: int | list[int] | tuple[int, ...],
) -> list[int]:
    if isinstance(num_cells, int):
        return [num_cells for _ in range(depth)]
    hidden_sizes = list(num_cells)
    if len(hidden_sizes) != depth:
        raise ValueError(
            f"Expected {depth} hidden layers, got {len(hidden_sizes)} from num_cells={num_cells}."
        )
    return hidden_sizes


def _normalize_optional_cfg_value(value):
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return value


def configure_advanced_model_inputs(cfg: DictConfig) -> None:
    is_actor_retentive = bool(cfg.model.get("is_actor_retentive", False))
    is_critic_bi_head = bool(cfg.model.get("is_critic_bi_head", False))

    if is_actor_retentive:
        if not bool(cfg.env.scenario.get("use_history_observation", False)):
            torchrl_logger.warning(
                "Retentive actor requires raw history observations. "
                "Overriding env.scenario.use_history_observation=True."
            )
            cfg.env.scenario.use_history_observation = True
        if _normalize_optional_cfg_value(
            cfg.env.scenario.get("history_obs_dim", None)
        ) is not None:
            torchrl_logger.warning(
                "Retentive actor expects raw per-step observation features. "
                "Overriding env.scenario.history_obs_dim=None."
            )
            cfg.env.scenario.history_obs_dim = None

    if (is_actor_retentive or is_critic_bi_head) and bool(
        cfg.model.get("use_phase_conditioned_network", False)
    ):
        torchrl_logger.warning(
            "Retentive actor / phase-aware critic supersede the older "
            "phase-conditioned MLP path. Overriding model.use_phase_conditioned_network=False."
        )
        cfg.model.use_phase_conditioned_network = False

    if is_critic_bi_head and not bool(cfg.model.get("centralised_critic", True)):
        torchrl_logger.warning(
            "Phase-aware critic requires a centralized critic. "
            "Overriding model.centralised_critic=True."
        )
        cfg.model.centralised_critic = True


def maybe_reverse_history_sequence(
    obs: torch.Tensor, *, latest_index: int = 0, time_dim: int = -2
) -> torch.Tensor:
    if obs.shape[time_dim] <= 1:
        return obs
    if latest_index == 0:
        return torch.flip(obs, dims=[time_dim])
    return obs


def _resolve_history_index(size: int, latest_index: int) -> int:
    resolved_index = latest_index if latest_index >= 0 else size + latest_index
    if not 0 <= resolved_index < size:
        raise IndexError(
            f"Resolved latest index {resolved_index} is out of range for history size {size}."
        )
    return resolved_index


def _ordered_named_observation_keys(obs_sample: TensorDictBase) -> list[str]:
    return [key for key in obs_sample.keys() if isinstance(key, str)]


def _normalize_key_list_cfg(value) -> list[str] | None:
    value = _normalize_optional_cfg_value(value)
    if value is None:
        return None
    return [str(item) for item in value]


def resolve_model_obs_keys(
    *,
    available_keys: list[str],
    include_keys,
    exclude_keys,
    label: str,
) -> list[str]:
    include_keys = _normalize_key_list_cfg(include_keys)
    exclude_keys = set(_normalize_key_list_cfg(exclude_keys) or [])
    available_key_set = set(available_keys)

    if include_keys is None:
        resolved_keys = [key for key in available_keys if key not in exclude_keys]
    else:
        missing_keys = [key for key in include_keys if key not in available_key_set]
        if missing_keys:
            raise KeyError(
                f"{label} requested unavailable observation keys: {missing_keys}. "
                f"Available keys: {available_keys}"
            )
        resolved_keys = [key for key in include_keys if key not in exclude_keys]

    if not resolved_keys:
        raise ValueError(f"{label} resolved to an empty observation-key list.")
    return resolved_keys


def resolve_model_phase_key(candidate_key: str, available_keys: list[str], label: str) -> str:
    if candidate_key not in available_keys:
        raise KeyError(
            f"{label}='{candidate_key}' is unavailable. Available observation keys: {available_keys}"
        )
    return candidate_key


def _select_latest_named_leaf(
    value: torch.Tensor,
    *,
    batch_dims: int,
    latest_history_index: int,
    use_history_observation: bool,
) -> torch.Tensor:
    if not use_history_observation or value.ndim <= batch_dims + 1:
        return value
    resolved_index = _resolve_history_index(
        value.shape[batch_dims], latest_history_index
    )
    return value.select(batch_dims, resolved_index)


def _flatten_named_leaf(value: torch.Tensor, *, batch_dims: int) -> torch.Tensor:
    if value.ndim <= batch_dims:
        return value.unsqueeze(-1)
    return value.flatten(batch_dims, -1)


def _flatten_named_leaf_preserve_time(
    value: torch.Tensor, *, batch_dims: int
) -> torch.Tensor:
    if value.ndim <= batch_dims + 1:
        return value.unsqueeze(-1)
    return value.flatten(batch_dims + 1, -1)


def infer_named_observation_layout(
    obs_sample: TensorDictBase,
    *,
    use_history_observation: bool,
    latest_history_index: int,
) -> dict[str, object]:
    obs_keys = _ordered_named_observation_keys(obs_sample)
    current_flat_dims = {}
    current_input_dim = 0
    for key in obs_keys:
        value = obs_sample.get(key)
        current_value = _select_latest_named_leaf(
            value,
            batch_dims=obs_sample.batch_dims,
            latest_history_index=latest_history_index,
            use_history_observation=use_history_observation,
        )
        flat_dim = int(_flatten_named_leaf(current_value, batch_dims=obs_sample.batch_dims).shape[-1])
        current_flat_dims[key] = flat_dim
        current_input_dim += flat_dim
    return {
        "obs_keys": obs_keys,
        "current_flat_dims": current_flat_dims,
        "current_input_dim": current_input_dim,
    }


def _move_nested_tensordict_to_device(
    td: TensorDictBase,
    key,
    device: torch.device | str,
) -> None:
    nested = _safe_get(td, key)
    if isinstance(nested, TensorDictBase):
        td.set(key, nested.to(device))


def _log_named_observation_devices(
    td: TensorDictBase,
    key,
    *,
    label: str,
) -> None:
    nested = _safe_get(td, key)
    if not isinstance(nested, TensorDictBase):
        print(f"[OBS_DEVICE][{label}] key={key} missing or not a TensorDict.")
        return
    parts = []
    for obs_key in nested.keys():
        value = nested.get(obs_key)
        if isinstance(value, torch.Tensor):
            parts.append(f"{obs_key}:device={value.device},shape={tuple(value.shape)}")
    print(f"[OBS_DEVICE][{label}] " + " | ".join(parts))


def _log_module_parameter_device(module: nn.Module, label: str) -> None:
    try:
        first_param = next(module.parameters())
    except StopIteration:
        print(f"[MODULE_DEVICE][{label}] no parameters")
        return
    print(f"[MODULE_DEVICE][{label}] device={first_param.device}")


class NamedObservationProjector(nn.Module):
    def __init__(
        self,
        *,
        obs_keys: list[str],
        use_history_observation: bool,
        latest_history_index: int,
    ) -> None:
        super().__init__()
        self.obs_keys = obs_keys
        self.use_history_observation = use_history_observation
        self.latest_history_index = latest_history_index

    def forward(self, obs: TensorDictBase | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            if self.use_history_observation:
                resolved_index = _resolve_history_index(obs.shape[-2], self.latest_history_index)
                obs = obs.select(-2, resolved_index)
            return obs.flatten(-1, -1) if obs.ndim <= 3 else obs.flatten(-2, -1)

        batch_dims = obs.batch_dims
        flat_values = []
        for key in self.obs_keys:
            value = obs.get(key)
            value = _select_latest_named_leaf(
                value,
                batch_dims=batch_dims,
                latest_history_index=self.latest_history_index,
                use_history_observation=self.use_history_observation,
            )
            flat_values.append(_flatten_named_leaf(value, batch_dims=batch_dims))
        return torch.cat(flat_values, dim=-1)


class NamedPhaseExtractor(nn.Module):
    def __init__(
        self,
        *,
        phase_key: str,
        use_history_observation: bool,
        latest_history_index: int,
        phase_reduce: str = "max",
    ) -> None:
        super().__init__()
        self.phase_key = phase_key
        self.use_history_observation = use_history_observation
        self.latest_history_index = latest_history_index
        self.phase_reduce = phase_reduce

    def forward(self, obs: TensorDictBase) -> torch.Tensor:
        phase_value = obs.get(self.phase_key).to(torch.float32)
        phase_value = _select_latest_named_leaf(
            phase_value,
            batch_dims=obs.batch_dims,
            latest_history_index=self.latest_history_index,
            use_history_observation=self.use_history_observation,
        )
        phase_value = _flatten_named_leaf(phase_value, batch_dims=obs.batch_dims)
        if self.phase_reduce == "mean":
            phase_value = phase_value.mean(dim=-1, keepdim=True)
        elif self.phase_reduce == "max":
            phase_value = phase_value.amax(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unsupported phase_reduce '{self.phase_reduce}'.")
        return phase_value.clamp(0.0, 1.0)


class SharedTrunkBiHeadNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        out_features: int,
        hidden_sizes: list[int],
        head_hidden: int | None = None,
        activation_class: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_sizes[-1] if hidden_sizes else max(input_dim, out_features, 32)
        head_hidden = hidden_dim if head_hidden is None else head_hidden
        if hidden_sizes:
            self.trunk = build_mlp(
                input_dim,
                hidden_dim,
                hidden_sizes[:-1],
                activation_class=activation_class,
            )
        else:
            self.trunk = nn.Identity()
        self.platoon_head = build_mlp(
            hidden_dim,
            out_features,
            [head_hidden],
            activation_class=activation_class,
        )
        self.hinge_head = build_mlp(
            hidden_dim,
            out_features,
            [head_hidden],
            activation_class=activation_class,
        )

    def forward(self, inputs: torch.Tensor, phase_weight: torch.Tensor) -> torch.Tensor:
        features = self.trunk(inputs)
        platoon_output = self.platoon_head(features)
        hinge_output = self.hinge_head(features)
        while phase_weight.ndim < platoon_output.ndim:
            phase_weight = phase_weight.unsqueeze(-1)
        return (1.0 - phase_weight) * platoon_output + phase_weight * hinge_output


class NamedPhaseConditionedMultiAgentBackbone(nn.Module):
    def __init__(
        self,
        *,
        obs_keys: list[str],
        phase_key: str,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        depth: int,
        num_cells: int | list[int] | tuple[int, ...],
        head_hidden: int | None = None,
        activation_class: type[nn.Module] = nn.Tanh,
        use_history_observation: bool = False,
        latest_history_index: int = 0,
        phase_reduce: str = "max",
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_agent_outputs = n_agent_outputs
        self.centralised = centralised
        self.share_params = share_params
        self.projector = NamedObservationProjector(
            obs_keys=obs_keys,
            use_history_observation=use_history_observation,
            latest_history_index=latest_history_index,
        )
        self.phase_extractor = NamedPhaseExtractor(
            phase_key=phase_key,
            use_history_observation=use_history_observation,
            latest_history_index=latest_history_index,
            phase_reduce=phase_reduce,
        )
        hidden_sizes = resolve_hidden_sizes(depth, num_cells)
        input_dim = n_agent_inputs * n_agents if centralised else n_agent_inputs
        network_count = 1 if share_params else n_agents
        self.networks = nn.ModuleList(
            [
                SharedTrunkBiHeadNetwork(
                    input_dim=input_dim,
                    out_features=n_agent_outputs,
                    hidden_sizes=hidden_sizes,
                    head_hidden=head_hidden,
                    activation_class=activation_class,
                )
                for _ in range(network_count)
            ]
        )

    def forward(self, obs: TensorDictBase) -> torch.Tensor:
        flat_obs = self.projector(obs)
        phase_weight = self.phase_extractor(obs)
        shared_inputs = flat_obs.flatten(-2, -1) if self.centralised else flat_obs
        if self.centralised:
            phase_for_network = phase_weight.mean(dim=-2, keepdim=False).clamp(0.0, 1.0)
        else:
            phase_for_network = phase_weight

        if self.share_params:
            if self.centralised:
                output = self.networks[0](shared_inputs, phase_for_network)
                return output.unsqueeze(-2).expand(*output.shape[:-1], self.n_agents, self.n_agent_outputs)
            return self.networks[0](shared_inputs, phase_for_network)

        outputs = []
        for agent_idx, network in enumerate(self.networks):
            agent_inputs = shared_inputs if self.centralised else shared_inputs[..., agent_idx, :]
            agent_phase = (
                phase_for_network
                if self.centralised
                else phase_for_network[..., agent_idx, :]
            )
            outputs.append(network(agent_inputs, agent_phase).unsqueeze(-2))
        return torch.cat(outputs, dim=-2)


class NamedPhaseAwareCentralCritic(nn.Module):
    def __init__(
        self,
        *,
        obs_keys: list[str],
        phase_key: str,
        n_agent_inputs: int,
        n_agents: int,
        share_params: bool,
        depth: int,
        num_cells: int | list[int] | tuple[int, ...],
        activation_class: type[nn.Module] = nn.Tanh,
        use_history_observation: bool = False,
        latest_history_index: int = 0,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.share_params = share_params
        self.projector = NamedObservationProjector(
            obs_keys=obs_keys,
            use_history_observation=use_history_observation,
            latest_history_index=latest_history_index,
        )
        self.phase_extractor = NamedPhaseExtractor(
            phase_key=phase_key,
            use_history_observation=use_history_observation,
            latest_history_index=latest_history_index,
            phase_reduce="max",
        )
        hidden_sizes = resolve_hidden_sizes(depth, num_cells)
        input_dim = n_agents * n_agent_inputs + n_agents
        network_count = 1 if share_params else n_agents
        self.networks = nn.ModuleList(
            [
                SharedTrunkBiHeadNetwork(
                    input_dim=input_dim,
                    out_features=1,
                    hidden_sizes=hidden_sizes,
                    activation_class=activation_class,
                )
                for _ in range(network_count)
            ]
        )

    def forward(self, obs: TensorDictBase) -> torch.Tensor:
        flat_obs = self.projector(obs)
        phase_value = self.phase_extractor(obs)
        global_obs = flat_obs.flatten(-2, -1)
        global_phase = phase_value.reshape(*phase_value.shape[:-2], -1)
        critic_input = torch.cat([global_obs, global_phase], dim=-1)
        agent_phase = phase_value.reshape(*phase_value.shape[:-1]).clamp(0.0, 1.0)

        if self.share_params:
            outputs = []
            shared_network = self.networks[0]
            for agent_idx in range(self.n_agents):
                outputs.append(
                    shared_network(
                        critic_input,
                        agent_phase[..., agent_idx : agent_idx + 1],
                    ).unsqueeze(-2)
                )
            return torch.cat(outputs, dim=-2)

        outputs = []
        for agent_idx, network in enumerate(self.networks):
            outputs.append(
                network(
                    critic_input,
                    agent_phase[..., agent_idx : agent_idx + 1],
                ).unsqueeze(-2)
            )
        return torch.cat(outputs, dim=-2)


class NamedRetentiveActorAgent(nn.Module):
    def __init__(
        self,
        *,
        self_current_keys: list[str],
        hinge_history_key: str | None,
        other_history_keys: list[str],
        hidden_sizes: list[int],
        sample_agent_obs: dict[str, torch.Tensor],
        latest_history_index: int = 0,
        out_features: int,
        activation_class: type[nn.Module] = nn.Tanh,
        branch_hidden: int | None = None,  # 新增：可配置branch hidden size
        lstm_hidden: int | None = None,    # 新增：可配置LSTM hidden size
    ) -> None:
        super().__init__()
        self.self_current_keys = self_current_keys
        self.hinge_history_key = hinge_history_key
        self.other_history_keys = other_history_keys
        self.latest_history_index = latest_history_index

        # sample_agent_obs values keep the environment batch dimension after
        # selecting one representative agent, so history starts after 1 batch dim.
        batch_dims = 1
        self_current_dim = 0
        for key in self_current_keys:
            current_value = _select_latest_named_leaf(
                sample_agent_obs[key],
                batch_dims=batch_dims,
                latest_history_index=latest_history_index,
                use_history_observation=True,
            )
            self_current_dim += int(
                _flatten_named_leaf(current_value, batch_dims=batch_dims).shape[-1]
            )

        base_hidden = hidden_sizes[-1] if hidden_sizes else max(out_features, 64)
        # 如果yaml中配置了branch_hidden，使用配置值；否则使用默认值
        self.branch_hidden = branch_hidden if branch_hidden is not None else max(32, base_hidden // 2)
        self.lstm_hidden = lstm_hidden if lstm_hidden is not None else self.branch_hidden
        self.self_encoder = build_mlp(
            max(self_current_dim, 1),
            self.branch_hidden,
            [self.branch_hidden],
            activation_class=activation_class,
        )

        if hinge_history_key is not None:
            hinge_value = sample_agent_obs[hinge_history_key]
            if hinge_value.ndim != batch_dims + 2:
                raise RuntimeError(
                    f"Retentive hinge key '{hinge_history_key}' is expected to have shape "
                    f"(*, history_len, feat_dim), but got {tuple(hinge_value.shape)}."
                )
            hinge_step_dim = int(
                _flatten_named_leaf_preserve_time(
                    hinge_value, batch_dims=batch_dims
                ).shape[-1]
            )
            self.hinge_step_encoder = build_mlp(
                hinge_step_dim,
                self.branch_hidden,
                [self.branch_hidden],
                activation_class=activation_class,
            )
            self.hinge_lstm = nn.LSTM(
                input_size=self.branch_hidden,
                hidden_size=self.lstm_hidden,
                batch_first=True,
            )
            self.hinge_feature_dim = self.lstm_hidden
        else:
            self.hinge_step_encoder = None
            self.hinge_lstm = None
            self.hinge_feature_dim = 0

        if other_history_keys:
            first_other = sample_agent_obs[other_history_keys[0]]
            self.n_other_agents = int(first_other.shape[batch_dims + 1])
            other_entity_dim = 0
            for key in other_history_keys:
                other_value = sample_agent_obs[key]
                other_entity_dim += int(other_value.flatten(batch_dims + 2, -1).shape[-1])
            self.other_entity_encoder = build_mlp(
                other_entity_dim,
                self.branch_hidden,
                [self.branch_hidden],
                activation_class=activation_class,
            )
            self.other_step_projector = build_mlp(
                self.n_other_agents * self.branch_hidden,
                self.branch_hidden,
                [self.branch_hidden],
                activation_class=activation_class,
            )
            self.other_lstm = nn.LSTM(
                input_size=self.branch_hidden,
                hidden_size=self.lstm_hidden,
                batch_first=True,
            )
            self.other_feature_dim = self.lstm_hidden
        else:
            self.n_other_agents = 0
            self.other_entity_encoder = None
            self.other_step_projector = None
            self.other_lstm = None
            self.other_feature_dim = 0

        fusion_in_dim = branch_hidden + self.hinge_feature_dim + self.other_feature_dim
        self.fusion_mlp = build_mlp(
            fusion_in_dim,
            out_features,
            hidden_sizes,
            activation_class=activation_class,
        )

    def _assert_same_device(
        self, tensor: torch.Tensor, module: nn.Module, label: str
    ) -> None:
        try:
            param_device = next(module.parameters()).device
        except StopIteration:
            return
        if tensor.device != param_device:
            raise RuntimeError(
                f"[DEVICE_MISMATCH][{label}] input_device={tensor.device} "
                f"module_device={param_device} shape={tuple(tensor.shape)}"
            )

    def _self_feature(self, obs: dict[str, torch.Tensor], batch_dims: int) -> torch.Tensor:
        features = []
        for key in self.self_current_keys:
            current_value = _select_latest_named_leaf(
                obs[key],
                batch_dims=batch_dims,
                latest_history_index=self.latest_history_index,
                use_history_observation=True,
            )
            features.append(_flatten_named_leaf(current_value, batch_dims=batch_dims))
        self_feature = torch.cat(features, dim=-1)
        self._assert_same_device(self_feature, self.self_encoder, "self_encoder")
        return self.self_encoder(self_feature)

    def _hinge_feature(
        self, obs: dict[str, torch.Tensor], batch_dims: int
    ) -> torch.Tensor | None:
        if self.hinge_history_key is None or self.hinge_step_encoder is None:
            return None
        hinge_value = obs[self.hinge_history_key]
        if hinge_value.ndim != batch_dims + 2:
            raise RuntimeError(
                f"Retentive hinge key '{self.hinge_history_key}' is expected to have shape "
                f"(*, history_len, feat_dim), but got {tuple(hinge_value.shape)}."
            )
        hinge_value = maybe_reverse_history_sequence(
            hinge_value,
            latest_index=self.latest_history_index,
            time_dim=batch_dims,
        )
        hinge_steps = _flatten_named_leaf_preserve_time(
            hinge_value, batch_dims=batch_dims
        )
        self._assert_same_device(
            hinge_steps, self.hinge_step_encoder, "hinge_step_encoder"
        )
        hinge_encoded = self.hinge_step_encoder(hinge_steps)
        self._assert_same_device(hinge_encoded, self.hinge_lstm, "hinge_lstm")
        _, (hidden_state, _) = self.hinge_lstm(hinge_encoded)
        return hidden_state[-1]

    def _other_feature(
        self, obs: dict[str, torch.Tensor], batch_dims: int
    ) -> torch.Tensor | None:
        if not self.other_history_keys or self.other_entity_encoder is None:
            return None
        other_parts = []
        for key in self.other_history_keys:
            other_value = maybe_reverse_history_sequence(
                obs[key],
                latest_index=self.latest_history_index,
                time_dim=batch_dims,
            )
            if other_value.ndim == batch_dims + 2:
                other_value = other_value.unsqueeze(-1)
            else:
                other_value = other_value.flatten(batch_dims + 2, -1)
            other_parts.append(other_value)
        other_value = torch.cat(other_parts, dim=-1)
        self._assert_same_device(
            other_value, self.other_entity_encoder, "other_entity_encoder"
        )
        encoded_entities = self.other_entity_encoder(other_value)
        flattened_entities = encoded_entities.reshape(
            *encoded_entities.shape[: batch_dims + 1],
            self.n_other_agents * encoded_entities.shape[-1],
        )
        self._assert_same_device(
            flattened_entities, self.other_step_projector, "other_step_projector"
        )
        other_step_features = self.other_step_projector(flattened_entities)
        self._assert_same_device(other_step_features, self.other_lstm, "other_lstm")
        _, (hidden_state, _) = self.other_lstm(other_step_features)
        return hidden_state[-1]

    def forward(self, obs: dict[str, torch.Tensor], batch_dims: int) -> torch.Tensor:
        branch_features = [self._self_feature(obs, batch_dims)]
        hinge_feature = self._hinge_feature(obs, batch_dims)
        if hinge_feature is not None:
            branch_features.append(hinge_feature)
        other_feature = self._other_feature(obs, batch_dims)
        if other_feature is not None:
            branch_features.append(other_feature)
        return self.fusion_mlp(torch.cat(branch_features, dim=-1))


class NamedRetentiveActorBackbone(nn.Module):
    def __init__(
        self,
        *,
        sample_obs: TensorDictBase,
        self_current_keys: list[str],
        hinge_history_key: str | None,
        other_history_keys: list[str],
        n_agent_outputs: int,
        n_agents: int,
        share_params: bool,
        depth: int,
        num_cells: int | list[int] | tuple[int, ...],
        activation_class: type[nn.Module] = nn.Tanh,
        latest_history_index: int = 0,
        branch_hidden: int | None = None,  # 新增：可配置branch hidden size
        lstm_hidden: int | None = None,    # 新增：可配置LSTM hidden size
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        self.self_current_keys = self_current_keys
        self.hinge_history_key = hinge_history_key
        self.other_history_keys = other_history_keys
        hidden_sizes = resolve_hidden_sizes(depth, num_cells)
        selected_keys = set(self_current_keys + other_history_keys)
        if hinge_history_key is not None:
            selected_keys.add(hinge_history_key)
        sample_agent_obs = {
            key: sample_obs.get(key).select(sample_obs.batch_dims - 1, 0)
            for key in selected_keys
        }
        network_count = 1 if share_params else n_agents
        self.agent_networks = nn.ModuleList(
            [
                NamedRetentiveActorAgent(
                    self_current_keys=self_current_keys,
                    hinge_history_key=hinge_history_key,
                    other_history_keys=other_history_keys,
                    hidden_sizes=hidden_sizes,
                    sample_agent_obs=sample_agent_obs,
                    latest_history_index=latest_history_index,
                    out_features=n_agent_outputs,
                    activation_class=activation_class,
                    branch_hidden=branch_hidden,  # 新增：传递参数
                    lstm_hidden=lstm_hidden,      # 新增：传递参数
                )
                for _ in range(network_count)
            ]
        )

    def forward(self, obs: TensorDictBase) -> torch.Tensor:
        agent_dim = obs.batch_dims - 1
        batch_dims_wo_agent = obs.batch_dims - 1
        outputs = []
        for agent_idx in range(self.n_agents):
            agent_obs = {
                key: obs.get(key).select(agent_dim, agent_idx)
                for key in self.agent_networks[0].self_current_keys
            }
            if self.hinge_history_key is not None:
                agent_obs[self.hinge_history_key] = obs.get(self.hinge_history_key).select(
                    agent_dim, agent_idx
                )
            for key in self.other_history_keys:
                agent_obs[key] = obs.get(key).select(agent_dim, agent_idx)
            network = self.agent_networks[0] if self.share_params else self.agent_networks[agent_idx]
            outputs.append(
                network(agent_obs, batch_dims_wo_agent).unsqueeze(-2)
            )
        return torch.cat(outputs, dim=-2)


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

@hydra.main(version_base="1.1", config_path="config/occt_extend", config_name="mappo_road_extend_baseline")
def train(cfg: DictConfig):
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.max_steps = eval(cfg.env.max_steps)
    configure_advanced_model_inputs(cfg)
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

    share_params = bool(cfg.model.shared_parameters)
    is_actor_retentive = bool(cfg.model.get("is_actor_retentive", False))
    is_critic_bi_head = bool(cfg.model.get("is_critic_bi_head", False))
    use_phase_conditioned_network = bool(
        cfg.model.get("use_phase_conditioned_network", False)
    )
    history_latest_index = int(cfg.model.get("history_latest_index", 0))
    actor_cfg = cfg.model.actor
    critic_cfg = cfg.model.critic
    actor_fields_cfg = actor_cfg.fields
    critic_fields_cfg = critic_cfg.fields
    actor_retentive_cfg = actor_cfg.get("retentive", {})
    # Policy
    actor_depth = int(actor_cfg.depth)
    actor_num_cells = actor_cfg.num_cells
    critic_depth = int(critic_cfg.depth)
    critic_num_cells = critic_cfg.num_cells
    critic_head_hidden = critic_cfg.get("head_hidden", None)

    uses_history_observation = bool(cfg.env.scenario.get("use_history_observation", False))
    action_dim = env.full_action_spec_unbatched[env.action_key].shape[-1]
    sample_td = env.reset()
    sample_obs = sample_td.get(("agents", "observation"))
    if not isinstance(sample_obs, TensorDictBase):
        raise TypeError(
            "OCCT training now expects dictionary-based observations from the scenario. "
            f"Got observation type {type(sample_obs).__name__} instead."
        )
    if uses_history_observation and not is_actor_retentive:
        raise RuntimeError(
            "is_actor_retentive=False does not allow multi-frame history observations. "
            "Disable env.scenario.use_history_observation or enable model.is_actor_retentive."
        )
    obs_layout = infer_named_observation_layout(
        sample_obs,
        use_history_observation=uses_history_observation,
        latest_history_index=history_latest_index,
    )
    available_obs_keys = list(obs_layout["obs_keys"])
    obs_key_dims = dict(obs_layout["current_flat_dims"])

    actor_obs_keys = resolve_model_obs_keys(
        available_keys=available_obs_keys,
        include_keys=actor_fields_cfg.get("include", None),
        exclude_keys=actor_fields_cfg.get("exclude", None),
        label="actor_obs_keys",
    )
    critic_obs_keys = resolve_model_obs_keys(
        available_keys=available_obs_keys,
        include_keys=critic_fields_cfg.get("include", None),
        exclude_keys=critic_fields_cfg.get("exclude", None),
        label="critic_obs_keys",
    )
    actor_input_dim = sum(obs_key_dims[key] for key in actor_obs_keys)
    critic_input_dim = sum(obs_key_dims[key] for key in critic_obs_keys)

    if use_phase_conditioned_network and not bool(cfg.model.get("centralised_critic", True)):
        torchrl_logger.warning(
            "Phase-conditioned OCCT training always uses a centralized critic. "
            "Overriding model.centralised_critic=False."
        )
    if use_phase_conditioned_network:
        torchrl_logger.warning(
            "model.use_phase_conditioned_network now affects only the critic path. "
            "The actor always uses either the standard single-head MLP or the retentive actor."
        )
    phase_obs_group = None
    if use_phase_conditioned_network or is_critic_bi_head:
        phase_obs_key = critic_cfg.get("phase_obs_key", None)
        if phase_obs_key is None:
            raise ValueError(
                "critic.phase_obs_key must be set when "
                "use_phase_conditioned_network=True or is_critic_bi_head=True."
            )
        phase_obs_group = resolve_model_phase_key(
            str(phase_obs_key),
            available_obs_keys,
            "critic.phase_obs_key",
        )

    if is_actor_retentive:
        retentive_self_keys = resolve_model_obs_keys(
            available_keys=available_obs_keys,
            include_keys=actor_retentive_cfg.get(
                "self_keys",
                [
                    "self_vel",
                    "self_speed",
                    "self_steering",
                    "self_acc",
                    "self_ref_velocity",
                    "self_ref_points",
                    "self_hinge_preview_info",
                    "self_distance_to_ref",
                    "self_left_boundary_distance",
                    "self_right_boundary_distance",
                    "self_distance_to_left_boundary",
                    "self_distance_to_right_boundary",
                    "self_platoon_error_vel",
                    "self_hinge_error_vel",
                    "self_platoon_error_space",
                ],
            ),
            exclude_keys=[],
            label="actor.retentive.self_keys",
        )
        retentive_other_keys = resolve_model_obs_keys(
            available_keys=available_obs_keys,
            include_keys=actor_retentive_cfg.get(
                "other_keys",
                [
                    "others_pos",
                    "others_rot",
                    "others_relative_longitudinal_velocity",
                    "others_distance",
                ],
            ),
            exclude_keys=[],
            label="actor.retentive.other_keys",
        )
        hinge_history_key = _normalize_optional_cfg_value(
            actor_retentive_cfg.get("hinge_key", "self_hinge_past_info")
        )
        if hinge_history_key is not None:
            hinge_history_key = resolve_model_phase_key(
                str(hinge_history_key),
                available_obs_keys,
                "actor.retentive.hinge_key",
            )
        # 从yaml配置读取branch_hidden和lstm_hidden参数
        retentive_cfg = actor_cfg.get("retentive", {})
        actor_branch_hidden = retentive_cfg.get("branch_hidden", None)
        actor_lstm_hidden = retentive_cfg.get("lstm_hidden", None)

        actor_backbone = NamedRetentiveActorBackbone(
            sample_obs=sample_obs,
            self_current_keys=retentive_self_keys,
            hinge_history_key=hinge_history_key,
            other_history_keys=retentive_other_keys,
            n_agent_outputs=2 * action_dim,
            n_agents=env.n_agents,
            share_params=share_params,
            depth=actor_depth,
            num_cells=actor_num_cells,
            activation_class=nn.Tanh,
            latest_history_index=history_latest_index,
            branch_hidden=actor_branch_hidden,  # 新增：从yaml读取
            lstm_hidden=actor_lstm_hidden,      # 新增：从yaml读取
        )
    else:
        actor_backbone = nn.Sequential(
            NamedObservationProjector(
                obs_keys=actor_obs_keys,
                use_history_observation=uses_history_observation,
                latest_history_index=history_latest_index,
            ),
            MultiAgentMLP(
                n_agent_inputs=actor_input_dim,
                n_agent_outputs=2 * action_dim,
                n_agents=env.n_agents,
                centralised=False,
                share_params=share_params,
                device=cfg.train.device,
                depth=actor_depth,
                num_cells=actor_num_cells,
                activation_class=nn.Tanh,
            ),
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
    policy = policy.to(cfg.train.device)
    _log_module_parameter_device(policy, "policy")

    # Critic
    if is_critic_bi_head:
        module = NamedPhaseAwareCentralCritic(
            obs_keys=critic_obs_keys,
            phase_key=phase_obs_group,
            n_agent_inputs=critic_input_dim,
            n_agents=env.n_agents,
            share_params=share_params,
            depth=critic_depth,
            num_cells=critic_num_cells,
            activation_class=nn.Tanh,
            use_history_observation=uses_history_observation,
            latest_history_index=history_latest_index,
        )
        value_module = ValueOperator(
            module=module,
            in_keys=[("agents", "observation")],
        )
    elif use_phase_conditioned_network:
        module = NamedPhaseConditionedMultiAgentBackbone(
            obs_keys=critic_obs_keys,
            phase_key=phase_obs_group,
            n_agent_inputs=critic_input_dim,
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralised=True,
            share_params=share_params,
            depth=critic_depth,
            num_cells=critic_num_cells,
            head_hidden=critic_head_hidden,
            activation_class=nn.Tanh,
            use_history_observation=uses_history_observation,
            latest_history_index=history_latest_index,
        )
        value_module = ValueOperator(
            module=module,
            in_keys=[("agents", "observation")],
        )
    else:
        module = nn.Sequential(
            NamedObservationProjector(
                obs_keys=critic_obs_keys,
                use_history_observation=uses_history_observation,
                latest_history_index=history_latest_index,
            ),
            MultiAgentMLP(
                n_agent_inputs=critic_input_dim,
                n_agent_outputs=1,
                n_agents=env.n_agents,
                centralised=cfg.model.centralised_critic,
                share_params=share_params,
                device=cfg.train.device,
                depth=critic_depth,
                num_cells=critic_num_cells,
                activation_class=nn.Tanh,
            ),
        )
        value_module = ValueOperator(
            module=module,
            in_keys=[("agents", "observation")],
        )
    value_module = value_module.to(cfg.train.device)
    _log_module_parameter_device(value_module, "value_module")
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
    trainable_params = [param for param in loss_module.parameters() if param.requires_grad]

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_iteration, start_frames = load_checkpoint(
            resume_from_checkpoint,
            policy=policy,
            value_module=value_module,
            optim=optim,
            flexible=False,
            resume_mode=resume_mode,
        )
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
        if is_actor_retentive and is_critic_bi_head:
            model_prefix = "RetentiveBiHeadMAPPO"
        elif is_actor_retentive:
            model_prefix = "RetentiveMAPPO"
        elif is_critic_bi_head:
            model_prefix = "BiHeadMAPPO"
        elif use_phase_conditioned_network:
            model_prefix = "PhaseMAPPO"
        else:
            model_prefix = "MAPPO"
        if (
            not use_phase_conditioned_network
            and not is_critic_bi_head
            and not cfg.model.centralised_critic
        ):
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
        iteration_extra_metrics = (
            phase_state["metrics"] if phase_weight_controller.enabled else None
        )
        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for epoch_idx in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                pbar.set_postfix({"Epoch": f"{epoch_idx+1}/{cfg.train.num_epochs}"})
                subdata = replay_buffer.sample().to(cfg.train.device)
                _move_nested_tensordict_to_device(
                    subdata, ("agents", "observation"), cfg.train.device
                )
                _move_nested_tensordict_to_device(
                    subdata, ("next", "agents", "observation"), cfg.train.device
                )
                if not minibatch_layout_logged:
                    _log_named_observation_devices(
                        subdata,
                        ("agents", "observation"),
                        label="minibatch_agents_observation",
                    )
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

                total_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                    trainable_params, cfg.train.max_grad_norm
                )
                total_norm_after_clip = _compute_total_grad_norm(trainable_params)
                max_grad_norm = float(cfg.train.max_grad_norm)
                assert total_norm_after_clip.item() <= max_grad_norm + 1e-4, (
                    "Gradient clipping did not bound the post-clip norm as expected: "
                    f"post_clip={total_norm_after_clip.item():.6f}, "
                    f"limit={max_grad_norm:.6f}"
                )
                training_tds[-1].set("grad_norm", total_norm_after_clip.detach())
                training_tds[-1].set(
                    "grad_norm_before_clip", total_norm_before_clip.detach()
                )

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
