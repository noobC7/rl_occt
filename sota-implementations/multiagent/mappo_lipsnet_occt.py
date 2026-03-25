import math
import os
import time

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

try:
    from torch.func import jacrev, vmap
except ImportError:
    from functorch import jacrev, vmap

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
from torchrl.modules.models.multiagent import MultiAgentMLP
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from mappo_ippo_occt import (
    MetricAdaptiveWeightController,
    build_eval_env,
    build_training_summary,
    configure_eval_env_paths,
    extract_training_phase_masks,
    infer_agent_advantage_exclude_dims,
    log_advantage_layout,
    rendering_callback,
    run_eval_export_chunk,
)
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform, load_checkpoint, save_checkpoint, save_rollout


def build_mlp(
    sizes: list[int],
    activation_class: type[nn.Module],
    output_activation_class: type[nn.Module] | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 2):
        linear = nn.Linear(sizes[idx], sizes[idx + 1])
        layers.extend((linear, activation_class()))
    final_linear = nn.Linear(sizes[-2], sizes[-1])
    layers.append(final_linear)
    if output_activation_class is not None:
        layers.append(output_activation_class())
    net = nn.Sequential(*layers)

    for idx, module in enumerate(net):
        if not isinstance(module, nn.Linear):
            continue
        next_module = net[idx + 1] if idx + 1 < len(net) else None
        if isinstance(next_module, nn.ReLU):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        elif isinstance(next_module, nn.LeakyReLU):
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
        else:
            nn.init.xavier_normal_(module.weight)
        nn.init.zeros_(module.bias)
    return net


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


def maybe_int(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return int(value)


class LipsNetAgent(nn.Module):
    def __init__(
        self,
        *,
        obs_len: int,
        obs_dim: int,
        out_features: int,
        hidden_sizes: list[int],
        activation_class: type[nn.Module] = nn.Tanh,
        lambda_t: float = 0.1,
        lambda_k: float = 0.0,
        kernel_scale: float = 0.02,
        norm_layer_type: str = "none",
        jacobian_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.obs_dim = obs_dim
        self.lambda_t = float(lambda_t)
        self.lambda_k = float(lambda_k)
        self.jacobian_samples = jacobian_samples
        self.norm_layer_type = norm_layer_type

        if norm_layer_type == "batch_norm":
            self.norm_layer = nn.BatchNorm1d(obs_dim)
        elif norm_layer_type == "layer_norm":
            self.norm_layer = nn.LayerNorm(obs_dim)
        elif norm_layer_type == "none":
            self.norm_layer = None
        else:
            raise ValueError(
                f"Unsupported LipsNet norm_layer_type='{norm_layer_type}'."
            )

        self.filter_kernel = nn.Parameter(
            torch.cat(
                [
                    torch.ones(obs_len, obs_dim // 2 + 1, 1, dtype=torch.float32),
                    torch.randn(obs_len, obs_dim // 2 + 1, 1, dtype=torch.float32)
                    * kernel_scale,
                ],
                dim=2,
            )
        )
        mlp_sizes = [obs_dim, *hidden_sizes, out_features]
        self.mlp = build_mlp(
            mlp_sizes,
            activation_class=activation_class,
            output_activation_class=None,
        )
        self._clear_last_stats()

    def _clear_last_stats(self) -> None:
        self._last_regularization = None
        self._last_filter_penalty = None
        self._last_jacobian_penalty = None
        self._last_jacobian_norm = None

    def _zero_scalar(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(())

    def _normalize_history(self, history: torch.Tensor) -> torch.Tensor:
        if self.norm_layer is None:
            return history
        if self.norm_layer_type == "batch_norm":
            return self.norm_layer(history.reshape(-1, self.obs_dim)).reshape(history.shape)
        return self.norm_layer(history)

    def _controller_forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    def _compute_filter_penalty(self, ref: torch.Tensor) -> torch.Tensor:
        if self.lambda_t == 0.0:
            return self._zero_scalar(ref)
        return self.lambda_t * (self.filter_kernel.square().sum())

    def _compute_jacobian_penalty(
        self, x_result: torch.Tensor, ref: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.lambda_k == 0.0:
            zero = self._zero_scalar(ref)
            return zero, zero

        jacobian_inputs = x_result.detach()
        if (
            self.jacobian_samples is not None
            and jacobian_inputs.shape[0] > self.jacobian_samples
        ):
            sampled_indices = torch.randperm(
                jacobian_inputs.shape[0],
                device=jacobian_inputs.device,
            )[: self.jacobian_samples]
            jacobian_inputs = jacobian_inputs[sampled_indices]

        jacobian = vmap(jacrev(self._controller_forward))(jacobian_inputs)
        jacobian_norm = torch.norm(jacobian, 2, dim=(-2, -1)).mean()
        return self.lambda_k * jacobian_norm, jacobian_norm

    def forward(self, history_obs: torch.Tensor) -> torch.Tensor:
        if history_obs.shape[-2:] != (self.obs_len, self.obs_dim):
            raise ValueError(
                "LipsNetAgent expected per-agent history observations with shape "
                f"(*, {self.obs_len}, {self.obs_dim}), got {tuple(history_obs.shape)}."
            )

        flat_history = history_obs.reshape(-1, self.obs_len, self.obs_dim)
        flat_history = self._normalize_history(flat_history)
        history_fft = torch.fft.rfft2(
            flat_history,
            s=(self.obs_len, self.obs_dim),
            dim=(-2, -1),
            norm="ortho",
        )
        kernel = torch.view_as_complex(self.filter_kernel)
        filtered_history = torch.fft.irfft2(
            history_fft * kernel,
            s=(self.obs_len, self.obs_dim),
            dim=(-2, -1),
            norm="ortho",
        )
        filtered_features = filtered_history[..., 0, :]
        output = self.mlp(filtered_features).reshape(*history_obs.shape[:-2], -1)

        compute_regularization = (
            self.training
            and torch.is_grad_enabled()
            and output.requires_grad
        )
        if not compute_regularization:
            zero = self._zero_scalar(output)
            self._last_regularization = zero
            self._last_filter_penalty = zero
            self._last_jacobian_penalty = zero
            self._last_jacobian_norm = zero
            return output

        filter_penalty = self._compute_filter_penalty(output)
        jacobian_penalty, jacobian_norm = self._compute_jacobian_penalty(
            filtered_features,
            output,
        )
        self._last_filter_penalty = filter_penalty
        self._last_jacobian_penalty = jacobian_penalty
        self._last_jacobian_norm = jacobian_norm
        self._last_regularization = filter_penalty + jacobian_penalty
        return output

    def pop_regularization_stats(self) -> dict[str, torch.Tensor]:
        stats = {
            "regularization_loss": self._last_regularization,
            "filter_penalty": self._last_filter_penalty,
            "jacobian_penalty": self._last_jacobian_penalty,
            "jacobian_norm": self._last_jacobian_norm,
        }
        self._clear_last_stats()
        return stats


class LipsNetMultiAgentBackbone(nn.Module):
    def __init__(
        self,
        *,
        obs_len: int,
        obs_dim: int,
        n_agent_outputs: int,
        n_agents: int,
        share_params: bool,
        device: torch.device | str,
        depth: int,
        num_cells: int | list[int] | tuple[int, ...],
        activation_class: type[nn.Module] = nn.Tanh,
        lambda_t: float = 0.1,
        lambda_k: float = 0.0,
        kernel_scale: float = 0.02,
        norm_layer_type: str = "none",
        jacobian_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.obs_len = obs_len
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_agent_outputs = n_agent_outputs
        self.share_params = share_params
        hidden_sizes = resolve_hidden_sizes(depth, num_cells)
        network_count = 1 if share_params else n_agents
        self.agent_networks = nn.ModuleList(
            [
                LipsNetAgent(
                    obs_len=obs_len,
                    obs_dim=obs_dim,
                    out_features=n_agent_outputs,
                    hidden_sizes=hidden_sizes,
                    activation_class=activation_class,
                    lambda_t=lambda_t,
                    lambda_k=lambda_k,
                    kernel_scale=kernel_scale,
                    norm_layer_type=norm_layer_type,
                    jacobian_samples=jacobian_samples,
                ).to(device)
                for _ in range(network_count)
            ]
        )
        self._clear_last_stats()

    def _clear_last_stats(self) -> None:
        self._last_regularization = None
        self._last_filter_penalty = None
        self._last_jacobian_penalty = None
        self._last_jacobian_norm = None

    def _zero_scalar(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.new_zeros(())

    def _collect_regularization_stats(
        self, stats_list: list[dict[str, torch.Tensor]], ref: torch.Tensor
    ) -> None:
        if not stats_list:
            zero = self._zero_scalar(ref)
            self._last_regularization = zero
            self._last_filter_penalty = zero
            self._last_jacobian_penalty = zero
            self._last_jacobian_norm = zero
            return

        def _average(key: str) -> torch.Tensor:
            values = [stats[key] for stats in stats_list if stats[key] is not None]
            if not values:
                return self._zero_scalar(ref)
            return torch.stack(values).mean()

        self._last_regularization = _average("regularization_loss")
        self._last_filter_penalty = _average("filter_penalty")
        self._last_jacobian_penalty = _average("jacobian_penalty")
        self._last_jacobian_norm = _average("jacobian_norm")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-3] != self.n_agents:
            raise ValueError(
                f"LipsNetMultiAgentBackbone expected obs.shape[-3] == {self.n_agents}, got {tuple(obs.shape)}."
            )
        if obs.shape[-2:] != (self.obs_len, self.obs_dim):
            raise ValueError(
                "LipsNetMultiAgentBackbone expected history observations with shape "
                f"(*, {self.n_agents}, {self.obs_len}, {self.obs_dim}), got {tuple(obs.shape)}."
            )

        if self.share_params:
            # Flatten batch and agent axes only to vectorize the same decentralized
            # per-agent policy. Each slice remains one agent's own [obs_len, obs_dim]
            # history; no cross-agent features are concatenated here.
            shared_output = self.agent_networks[0](
                obs.reshape(-1, self.obs_len, self.obs_dim)
            )
            output = shared_output.reshape(*obs.shape[:-2], self.n_agent_outputs)
            stats_list = [self.agent_networks[0].pop_regularization_stats()]
            self._collect_regularization_stats(stats_list, output)
            return output

        outputs = []
        stats_list = []
        for agent_idx, agent_network in enumerate(self.agent_networks):
            # Heterogeneous setting: each agent still consumes only its own history.
            agent_output = agent_network(obs[..., agent_idx, :, :])
            outputs.append(agent_output.unsqueeze(-2))
            stats_list.append(agent_network.pop_regularization_stats())
        output = torch.cat(outputs, dim=-2)
        self._collect_regularization_stats(stats_list, output)
        return output

    def pop_regularization_stats(self) -> dict[str, torch.Tensor]:
        stats = {
            "regularization_loss": self._last_regularization,
            "filter_penalty": self._last_filter_penalty,
            "jacobian_penalty": self._last_jacobian_penalty,
            "jacobian_norm": self._last_jacobian_norm,
        }
        self._clear_last_stats()
        return stats


class LipsNetActorModule(nn.Module):
    in_keys = [("agents", "observation")]
    out_keys = [("agents", "loc"), ("agents", "scale")]

    def __init__(self, backbone: LipsNetMultiAgentBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.param_extractor = NormalParamExtractor()

    def forward(self, tensordict):
        obs = tensordict.get(("agents", "observation"))
        params = self.backbone(obs)
        loc, scale = self.param_extractor(params)
        tensordict.set(("agents", "loc"), loc)
        tensordict.set(("agents", "scale"), scale)
        return tensordict

    def pop_regularization_stats(self) -> dict[str, torch.Tensor]:
        return self.backbone.pop_regularization_stats()


class FlattenHistoryValueWrapper(nn.Module):
    def __init__(self, base_module: nn.Module) -> None:
        super().__init__()
        self.base_module = base_module

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim >= 4:
            obs = obs.flatten(-2, -1)
        return self.base_module(obs)


def configure_history_observation(cfg: DictConfig) -> None:
    if bool(cfg.model.get("use_phase_conditioned_network", False)):
        torchrl_logger.warning(
            "mappo_lipsnet_occt does not currently support phase-conditioned networks "
            "with history observations. Overriding model.use_phase_conditioned_network=False."
        )
        cfg.model.use_phase_conditioned_network = False


@hydra.main(version_base="1.1", config_path="config", config_name="mappo_lipsnet_occt")
def train(cfg: DictConfig):
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.max_steps = eval(cfg.env.max_steps)
    configure_history_observation(cfg)
    print(cfg.env)
    torch.manual_seed(cfg.seed)

    resume_from_checkpoint = cfg.train.resume_from_checkpoint
    resume_mode = cfg.train.resume_mode
    start_iteration = 0
    start_frames = 0

    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch

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
    actor_depth = int(cfg.model.get("actor_depth", 2))
    actor_num_cells = cfg.model.get("actor_num_cells", 344)
    critic_depth = int(cfg.model.get("critic_depth", 2))
    critic_num_cells = cfg.model.get("critic_num_cells", 688)
    jacobian_samples = maybe_int(cfg.model.get("lipsnet_jacobian_samples", None))

    obs_spec_shape = tuple(
        env.full_observation_spec_unbatched["agents", "observation"].shape
    )
    if len(obs_spec_shape) < 3:
        raise RuntimeError(
            "LipsNet MAPPO expects per-agent history observations shaped as "
            f"[n_agents, obs_len, obs_dim], got observation spec shape {obs_spec_shape}."
        )
    per_agent_obs_shape = obs_spec_shape[1:]
    if len(per_agent_obs_shape) != 2:
        raise RuntimeError(
            "LipsNet actor expects exactly two per-agent observation dimensions "
            f"(obs_len, obs_dim), got {per_agent_obs_shape} from unbatched spec shape {obs_spec_shape}."
        )
    obs_len, obs_dim = map(int, per_agent_obs_shape)
    critic_input_dim = int(math.prod(per_agent_obs_shape))
    action_dim = env.full_action_spec_unbatched[env.action_key].shape[-1]

    actor_backbone = LipsNetMultiAgentBackbone(
        obs_len=obs_len,
        obs_dim=obs_dim,
        n_agent_outputs=2 * action_dim,
        n_agents=env.n_agents,
        share_params=share_params,
        device=cfg.train.device,
        depth=actor_depth,
        num_cells=actor_num_cells,
        activation_class=nn.Tanh,
        lambda_t=float(cfg.model.get("lipsnet_lambda_t", 0.1)),
        lambda_k=float(cfg.model.get("lipsnet_lambda_k", 0.0)),
        kernel_scale=float(cfg.model.get("lipsnet_kernel_scale", 0.02)),
        norm_layer_type=str(cfg.model.get("lipsnet_norm_layer_type", "none")),
        jacobian_samples=jacobian_samples,
    )
    actor_module = LipsNetActorModule(actor_backbone)
    policy = ProbabilisticActor(
        module=actor_module,
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

    critic_base = MultiAgentMLP(
        n_agent_inputs=critic_input_dim,
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=bool(cfg.model.centralised_critic),
        share_params=share_params,
        device=cfg.train.device,
        depth=critic_depth,
        num_cells=critic_num_cells,
        activation_class=nn.Tanh,
    )
    value_module = ValueOperator(
        module=FlattenHistoryValueWrapper(critic_base),
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

    if cfg.logger.backend:
        model_prefix = "LipsNetHet" if not cfg.model.shared_parameters else "LipsNet"
        model_name = model_prefix + ("MAPPO" if cfg.model.centralised_critic else "IPPO")
        logger = init_logging(cfg, model_name)

    total_time = 0.0
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

        for chunk_index, chunk_start in enumerate(
            range(0, total_eval_paths, render_batch_size)
        ):
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
                pbar.set_postfix({"Epoch": f"{epoch_idx + 1}/{cfg.train.num_epochs}"})
                subdata = replay_buffer.sample()
                if not minibatch_layout_logged:
                    log_advantage_layout("minibatch", subdata, loss_module, env.n_agents)
                    minibatch_layout_logged = True

                loss_vals = loss_module(subdata)
                lipsnet_stats = actor_module.pop_regularization_stats()

                platoon_mask, hinge_approach_mask = extract_training_phase_masks(subdata)
                active_phase_weights = (
                    phase_state["weights"] if phase_weight_controller.enabled else None
                )
                loss_value, training_summary = build_training_summary(
                    loss_vals,
                    platoon_mask,
                    hinge_approach_mask,
                    active_phase_weights,
                )

                lipsnet_regularization = lipsnet_stats["regularization_loss"]
                if lipsnet_regularization is not None:
                    loss_value = loss_value + lipsnet_regularization
                    training_summary.set(
                        "loss_lipsnet_regularization",
                        lipsnet_regularization.detach(),
                    )
                    if lipsnet_stats["filter_penalty"] is not None:
                        training_summary.set(
                            "loss_lipsnet_filter",
                            lipsnet_stats["filter_penalty"].detach(),
                        )
                    if lipsnet_stats["jacobian_penalty"] is not None:
                        training_summary.set(
                            "loss_lipsnet_jacobian",
                            lipsnet_stats["jacobian_penalty"].detach(),
                        )
                    if lipsnet_stats["jacobian_norm"] is not None:
                        training_summary.set(
                            "lipsnet_jacobian_norm",
                            lipsnet_stats["jacobian_norm"].detach(),
                        )
                training_summary.set(
                    "lipsnet_obs_len",
                    torch.tensor(float(obs_len), device=loss_value.device),
                )
                training_summary.set(
                    "lipsnet_obs_dim",
                    torch.tensor(float(obs_dim), device=loss_value.device),
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

        sampling_start = time.time()

    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()


if __name__ == "__main__":
    train()
