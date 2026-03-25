import os
import time

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
from torchrl.modules.models import MLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from mappo_ippo_occt import (
    AGENT_FOCUS_INDEX,
    build_eval_env,
    configure_eval_env_paths,
    infer_agent_advantage_exclude_dims,
    log_advantage_layout,
    rendering_callback,
    run_eval_export_chunk,
)
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform, load_checkpoint, save_checkpoint, save_rollout

DEFAULT_SOFT_NUM_MODULES = 2


def _make_linear(in_features: int, out_features: int, device=None) -> nn.Linear:
    if device is None:
        return nn.Linear(in_features, out_features)
    return nn.Linear(in_features, out_features, device=device)

def _safe_td_get(td, key):
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
    expanded_mask = _expand_mask(mask, value)
    if bool(expanded_mask.any().item()):
        return value[expanded_mask].mean()
    return value.sum() * 0.0


def _reduce_metric(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.mean()
    return torch.as_tensor(value)


def extract_phase_mask(td, phase_key: str = "hinge_status") -> torch.Tensor:
    phase_tensor = _safe_td_get(td, ("agents", "info", phase_key))
    if phase_tensor is None:
        phase_tensor = _safe_td_get(td, ("next", "agents", "info", phase_key))
    if phase_tensor is None:
        raise KeyError(
            f"Could not find {phase_key} in the sampled TensorDict. "
            "Adaptive phase weighting requires a valid phase indicator in env info."
        )
    return _ensure_phase_mask(phase_tensor)


class LinearInitArcCurriculum:
    def __init__(self, cfg: DictConfig, scenario) -> None:
        curriculum_cfg = cfg.get("curriculum", {})
        self.enabled = bool(curriculum_cfg.get("enabled", False))

        self.base_is_rand_arc_pos = bool(scenario.is_rand_arc_pos)
        self.base_init_arc_pos = float(scenario.init_arc_pos)
        self.base_init_vel_std = float(scenario.init_vel_std)

        self.start_iteration = int(curriculum_cfg.get("start_iteration", 0))
        self.end_iteration = int(curriculum_cfg.get("end_iteration", 0))
        self.curriculum_is_rand_arc_pos = bool(
            curriculum_cfg.get("is_rand_arc_pos", self.base_is_rand_arc_pos)
        )
        self.start_init_arc_pos = float(
            curriculum_cfg.get("start_init_arc_pos", self.base_init_arc_pos)
        )
        self.end_init_arc_pos = float(
            curriculum_cfg.get("end_init_arc_pos", self.base_init_arc_pos)
        )
        self.start_init_vel_std = float(
            curriculum_cfg.get("start_init_vel_std", self.base_init_vel_std)
        )
        self.end_init_vel_std = float(
            curriculum_cfg.get("end_init_vel_std", self.base_init_vel_std)
        )

    def _compute_progress(self, iteration: int) -> float:
        if not self.enabled:
            return 0.0
        if self.end_iteration <= self.start_iteration:
            return 1.0 if iteration >= self.end_iteration else 0.0
        progress = (iteration - self.start_iteration) / (
            self.end_iteration - self.start_iteration
        )
        return float(min(max(progress, 0.0), 1.0))

    def apply(self, scenario, iteration: int) -> dict[str, float]:
        progress = self._compute_progress(iteration)
        if not self.enabled:
            scenario.is_rand_arc_pos = self.base_is_rand_arc_pos
            scenario.init_arc_pos = self.base_init_arc_pos
            scenario.init_vel_std = self.base_init_vel_std
        else:
            scenario.is_rand_arc_pos = self.curriculum_is_rand_arc_pos
            scenario.init_arc_pos = (
                (1.0 - progress) * self.start_init_arc_pos
                + progress * self.end_init_arc_pos
            )
            scenario.init_vel_std = (
                (1.0 - progress) * self.start_init_vel_std
                + progress * self.end_init_vel_std
            )

        return {
            "train/curriculum/enabled": float(self.enabled),
            "train/curriculum/progress": progress,
            "train/curriculum/is_rand_arc_pos": float(scenario.is_rand_arc_pos),
            "train/curriculum/init_arc_pos": float(scenario.init_arc_pos),
            "train/curriculum/init_vel_std": float(scenario.init_vel_std),
        }


class PhaseAdaptiveWeightController:
    def __init__(self, cfg: DictConfig, device: torch.device | str) -> None:
        adaptive_cfg = cfg.get("adaptive_weighting", {})
        self.enabled = bool(adaptive_cfg.get("enabled", True))
        self.phase_key = adaptive_cfg.get("phase_key", "hinge_status")
        self.beta = float(adaptive_cfg.get("beta", 5.0))
        self.ema_tau = float(adaptive_cfg.get("ema_tau", 0.1))
        self.min_weight = float(adaptive_cfg.get("min_weight", 0.0))
        self.min_samples = int(adaptive_cfg.get("min_samples", 32))
        self.device = torch.device(device)

        self.platoon_entropy_ema = None
        self.hinge_entropy_ema = None

    def _make_weight_tensor(
        self, platoon_weight: float, hinge_weight: float, device: torch.device
    ) -> dict[str, torch.Tensor]:
        return {
            "platoon": torch.tensor(
                platoon_weight, device=device, dtype=torch.float32
            ),
            "hinge": torch.tensor(hinge_weight, device=device, dtype=torch.float32),
        }

    def _update_ema(self, current, ema_attr: str) -> torch.Tensor:
        previous = getattr(self, ema_attr)
        if previous is None:
            updated = current.detach()
        else:
            updated = torch.lerp(previous, current.detach(), self.ema_tau)
        setattr(self, ema_attr, updated)
        return updated

    def _collector_entropy(
        self, td, policy, loss_module, phase_mask: torch.Tensor
    ) -> torch.Tensor:
        policy_td = td.select(*policy.in_keys, strict=False)
        dist = policy.get_dist(policy_td)
        adv_shape = phase_mask.shape[:-1]
        return loss_module._get_entropy(dist, adv_shape=adv_shape).detach()

    def collector_state(self, td, policy, loss_module) -> dict[str, object]:
        phase_mask = extract_phase_mask(td, self.phase_key)
        hinge_count = int(phase_mask.sum().item())
        total_count = int(phase_mask.numel())
        platoon_count = total_count - hinge_count

        metrics = {
            "train/adaptive_weighting/enabled": float(self.enabled),
            "train/adaptive_weighting/collector_hinge_sample_count": float(
                hinge_count
            ),
            "train/adaptive_weighting/collector_platoon_sample_count": float(
                platoon_count
            ),
            "train/adaptive_weighting/collector_hinge_sample_ratio": (
                hinge_count / max(total_count, 1)
            ),
            "train/adaptive_weighting/collector_platoon_sample_ratio": (
                platoon_count / max(total_count, 1)
            ),
        }

        if not self.enabled:
            weights = self._make_weight_tensor(1.0, 0.0, phase_mask.device)
            metrics.update(
                {
                    "train/adaptive_weighting/w_platoon": 1.0,
                    "train/adaptive_weighting/w_hinge": 0.0,
                }
            )
            return {
                "weights": weights,
                "metrics": metrics,
            }

        with torch.no_grad():
            entropy = self._collector_entropy(td, policy, loss_module, phase_mask)

        platoon_mask = ~phase_mask
        entropy_platoon = _masked_tensor_mean(entropy, platoon_mask).detach()
        entropy_hinge = _masked_tensor_mean(entropy, phase_mask).detach()

        if platoon_count >= self.min_samples:
            platoon_ref = self._update_ema(entropy_platoon, "platoon_entropy_ema")
        else:
            platoon_ref = (
                self.platoon_entropy_ema
                if self.platoon_entropy_ema is not None
                else entropy_platoon
            )

        if hinge_count >= self.min_samples:
            hinge_ref = self._update_ema(entropy_hinge, "hinge_entropy_ema")
        else:
            hinge_ref = (
                self.hinge_entropy_ema
                if self.hinge_entropy_ema is not None
                else entropy_hinge
            )

        if hinge_count == 0:
            weights = self._make_weight_tensor(1.0, 0.0, phase_mask.device)
        elif platoon_count == 0:
            weights = self._make_weight_tensor(0.0, 1.0, phase_mask.device)
        else:
            logits = self.beta * torch.stack((platoon_ref, hinge_ref))
            weight_tensor = torch.softmax(logits, dim=0)
            if self.min_weight > 0.0:
                weight_tensor = torch.clamp(weight_tensor, min=self.min_weight)
                weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-8)
            weights = {
                "platoon": weight_tensor[0].detach(),
                "hinge": weight_tensor[1].detach(),
            }

        metrics.update(
            {
                "train/adaptive_weighting/collector_entropy_platoon": float(
                    entropy_platoon.item()
                ),
                "train/adaptive_weighting/collector_entropy_hinge": float(
                    entropy_hinge.item()
                ),
                "train/adaptive_weighting/ema_entropy_platoon": float(
                    platoon_ref.item()
                ),
                "train/adaptive_weighting/ema_entropy_hinge": float(hinge_ref.item()),
                "train/adaptive_weighting/w_platoon": float(
                    weights["platoon"].item()
                ),
                "train/adaptive_weighting/w_hinge": float(weights["hinge"].item()),
            }
        )

        return {
            "weights": weights,
            "metrics": metrics,
        }


def build_training_summary(
    loss_vals, phase_mask: torch.Tensor, phase_weights: dict[str, torch.Tensor] | None
) -> tuple[torch.Tensor, TensorDict]:
    summary = TensorDict({}, batch_size=[])
    phase_mask = _ensure_phase_mask(phase_mask)
    platoon_mask = ~phase_mask

    if phase_weights is None:
        component_totals = []
        for key in ("loss_objective", "loss_critic", "loss_entropy"):
            if key not in loss_vals.keys():
                continue
            component_value = _reduce_metric(loss_vals[key])
            summary.set(key, component_value.detach())
            component_totals.append(component_value)

        loss_value = sum(component_totals)
        summary.set("adaptive_weight_platoon", torch.tensor(1.0, device=loss_value.device))
        summary.set("adaptive_weight_hinge", torch.tensor(0.0, device=loss_value.device))
    else:
        phase_totals = {
            "platoon": torch.zeros((), device=phase_weights["platoon"].device),
            "hinge": torch.zeros((), device=phase_weights["hinge"].device),
        }
        weighted_component_totals = {}

        for key in ("loss_objective", "loss_critic", "loss_entropy"):
            if key not in loss_vals.keys():
                continue
            platoon_component = _masked_tensor_mean(loss_vals[key], platoon_mask)
            hinge_component = _masked_tensor_mean(loss_vals[key], phase_mask)

            phase_totals["platoon"] = phase_totals["platoon"] + platoon_component
            phase_totals["hinge"] = phase_totals["hinge"] + hinge_component

            weighted_component = (
                phase_weights["platoon"] * platoon_component
                + phase_weights["hinge"] * hinge_component
            )
            weighted_component_totals[key] = weighted_component
            summary.set(key, weighted_component.detach())
            summary.set(f"{key}_platoon", platoon_component.detach())
            summary.set(f"{key}_hinge", hinge_component.detach())

        loss_value = sum(weighted_component_totals.values())
        summary.set("adaptive_weight_platoon", phase_weights["platoon"].detach())
        summary.set("adaptive_weight_hinge", phase_weights["hinge"].detach())
        summary.set("adaptive_loss_platoon", phase_totals["platoon"].detach())
        summary.set("adaptive_loss_hinge", phase_totals["hinge"].detach())

    summary.set("adaptive_ratio_platoon", platoon_mask.float().mean().detach())
    summary.set("adaptive_ratio_hinge", phase_mask.float().mean().detach())

    for metric_key in ("entropy", "clip_fraction", "kl_approx", "ESS", "explained_variance"):
        metric_value = loss_vals.get(metric_key, None)
        if metric_value is not None:
            summary.set(metric_key, _reduce_metric(metric_value).detach())

    return loss_value, summary


class _SoftModularAgentMLP(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        out_features: int,
        device=None,
        depth: int | None = None,
        num_cells: int | list[int] | tuple[int, ...] | None = None,
        activation_class: type[nn.Module] = nn.Tanh,
        num_modules: int = DEFAULT_SOFT_NUM_MODULES,
        module_hidden: int | None = None,
    ):
        super().__init__()
        depth = 2 if depth is None else depth
        num_cells = 128 if num_cells is None else num_cells
        hidden_size = num_cells[-1] if isinstance(num_cells, (list, tuple)) else num_cells

        self.activation = activation_class()
        self.num_modules = num_modules
        self.module_hidden = hidden_size if module_hidden is None else module_hidden

        self.obs_encoder = MLP(
            in_features=input_dim,
            out_features=hidden_size,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            activate_last_layer=False,
            device=device,
        )
        self.gate_head = _make_linear(hidden_size, self.num_modules, device=device)
        self.experts = nn.ModuleList(
            [
                _make_linear(hidden_size, self.module_hidden, device=device)
                for _ in range(self.num_modules)
            ]
        )
        self.output_layer = _make_linear(self.module_hidden, out_features, device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        obs_features = self.obs_encoder(inputs)
        gate_weights = torch.softmax(self.gate_head(self.activation(obs_features)), dim=-1)
        expert_outputs = torch.stack(
            [expert(obs_features) for expert in self.experts], dim=-2
        )
        mixed_features = (expert_outputs * gate_weights.unsqueeze(-1)).sum(dim=-2)
        return self.output_layer(self.activation(mixed_features))


class SoftModularMultiAgentMLP(nn.Module):
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        *,
        centralized: bool | None = None,
        share_params: bool | None = None,
        device=None,
        depth: int | None = None,
        num_cells: int | list[int] | tuple[int, ...] | None = None,
        activation_class: type[nn.Module] = nn.Tanh,
        num_modules: int = DEFAULT_SOFT_NUM_MODULES,
        module_hidden: int | None = None,
        **kwargs,
    ):
        super().__init__()
        centralized = kwargs.pop("centralised", centralized)
        if centralized is None:
            raise TypeError("centralized arg must be passed.")
        if share_params is None:
            raise TypeError("share_params arg must be passed.")
        if n_agent_inputs is None:
            raise TypeError("n_agent_inputs must be passed.")

        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.centralized = centralized
        self.share_params = share_params

        input_dim = n_agent_inputs * n_agents if self.centralized else n_agent_inputs
        agent_network_count = 1 if self.share_params else self.n_agents
        self.agent_networks = nn.ModuleList(
            [
                _SoftModularAgentMLP(
                    input_dim=input_dim,
                    out_features=n_agent_outputs,
                    device=device,
                    depth=depth,
                    num_cells=num_cells,
                    activation_class=activation_class,
                    num_modules=num_modules,
                    module_hidden=module_hidden,
                )
                for _ in range(agent_network_count)
            ]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2] != self.n_agents:
            raise ValueError(
                f"Expected obs.shape[-2] == {self.n_agents}, got {obs.shape}."
            )
        if self.centralized:
            shared_inputs = obs.flatten(-2, -1)
        else:
            shared_inputs = obs

        if self.share_params:
            if self.centralized:
                output = self.agent_networks[0](shared_inputs)
                if output.shape[-1] != self.n_agent_outputs:
                    raise ValueError(
                        f"Unexpected centralized shared output shape {output.shape}."
                    )
                output = output.unsqueeze(-2).expand(
                    *output.shape[:-1], self.n_agents, self.n_agent_outputs
                )
                return output
            return self.agent_networks[0](shared_inputs)

        outputs = []
        for agent_idx, agent_network in enumerate(self.agent_networks):
            if self.centralized:
                agent_inputs = shared_inputs
            else:
                agent_inputs = shared_inputs[..., agent_idx, :]
            outputs.append(agent_network(agent_inputs).unsqueeze(-2))
        return torch.cat(outputs, dim=-2)


@hydra.main(version_base="1.1", config_path="config", config_name="mappo_occt_3_followers")
def train(cfg: DictConfig):
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.max_steps = eval(cfg.env.max_steps)
    torch.manual_seed(cfg.seed)
    resume_from_checkpoint = cfg.train.resume_from_checkpoint
    resume_mode = cfg.train.resume_mode
    start_iteration = 0
    start_frames = 0
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

    base_env = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.env.vmas_envs,
        continuous_actions=True,
        max_steps=cfg.env.max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario,
    )
    curriculum_controller = LinearInitArcCurriculum(cfg, base_env.scenario)
    curriculum_metrics = curriculum_controller.apply(base_env.scenario, start_iteration)
    phase_weight_controller = PhaseAdaptiveWeightController(cfg, cfg.train.device)
    actor_depth = int(cfg.model.get("actor_depth", 2))
    actor_num_cells = cfg.model.get("actor_num_cells", 344)
    actor_module_hidden = cfg.model.get("actor_module_hidden", 128)
    critic_depth = int(cfg.model.get("critic_depth", 2))
    critic_num_cells = cfg.model.get("critic_num_cells", 688)
    critic_module_hidden = cfg.model.get("critic_module_hidden", 256)
    soft_num_modules = int(cfg.model.get("num_modules", 2))
    if not cfg.model.shared_parameters:
        torchrl_logger.warning(
            "SoftModularMultiAgentMLP with shared_parameters=False builds one backbone per agent. "
            "This increases parameter count and activation memory roughly linearly with n_agents."
        )

    env = TransformedEnv(
        base_env,
        RewardSum(in_keys=[base_env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    cfg_test = cfg.copy()
    cfg_test.env.scenario.is_rand_arc_pos = False
    cfg_test.env.scenario.init_vel_std = 0
    env_test = None

    actor_backbone = SoftModularMultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2
        * env.full_action_spec_unbatched[env.action_key].shape[-1],
        n_agents=env.n_agents,
        centralised=False,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=actor_depth,
        num_cells=actor_num_cells,
        module_hidden=actor_module_hidden,
        activation_class=nn.Tanh,
        num_modules=soft_num_modules,
    )
    actor_net = nn.Sequential(
        actor_backbone,
        NormalParamExtractor(),
    )
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

    module = SoftModularMultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=True,
        share_params=cfg.model.shared_parameters,
        device=cfg.train.device,
        depth=critic_depth,
        num_cells=critic_num_cells,
        module_hidden=critic_module_hidden,
        activation_class=nn.Tanh,
        num_modules=soft_num_modules,
    )
    value_module = ValueOperator(
        module=module,
        in_keys=[("agents", "observation")],
    )

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

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_eps,
        normalize_advantage=True,
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

    if cfg.logger.backend:
        model_prefix = "SoftModHet" if not cfg.model.shared_parameters else "SoftMod"
        model_name = model_prefix + "MAPPO"
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
    pbar = tqdm(
        enumerate(collector, start=start_iteration),
        initial=start_iteration,
        total=cfg.collector.n_iters,
        desc="Training",
        unit="iter",
    )
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

        phase_state = phase_weight_controller.collector_state(
            tensordict_data,
            policy,
            loss_module,
        )
        iteration_extra_metrics = dict(curriculum_metrics)
        iteration_extra_metrics.update(phase_state["metrics"])

        current_frames = tensordict_data.numel()
        total_frames += current_frames
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        training_tds = []
        training_start = time.time()
        for epoch_idx in range(cfg.train.num_epochs):
            for _ in range(cfg.collector.frames_per_batch // cfg.train.minibatch_size):
                pbar.set_postfix({"Epoch": f"{epoch_idx + 1}/{cfg.train.num_epochs}"})
                subdata = replay_buffer.sample()
                if not minibatch_layout_logged:
                    log_advantage_layout("minibatch", subdata, loss_module, env.n_agents)
                    minibatch_layout_logged = True
                loss_vals = loss_module(subdata)
                phase_mask = extract_phase_mask(
                    subdata, phase_weight_controller.phase_key
                )
                active_phase_weights = (
                    phase_state["weights"] if phase_weight_controller.enabled else None
                )
                loss_value, training_summary = build_training_summary(
                    loss_vals,
                    phase_mask,
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
                )
                evaluation_time = time.time() - evaluation_start
                log_evaluation(
                    logger,
                    rollouts,
                    env_test,
                    evaluation_time,
                    step=i,
                    video_caption="path_0",
                )
                save_checkpoint(logger, policy, value_module, optim, i, total_frames)
                save_rollout(logger, rollouts, i, total_frames)

        curriculum_metrics = curriculum_controller.apply(base_env.scenario, i + 1)
        sampling_start = time.time()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()
