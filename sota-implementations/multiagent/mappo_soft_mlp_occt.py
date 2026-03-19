import os
import time

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from tqdm import tqdm

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
    rendering_callback,
    run_eval_export_chunk,
)
from utils.logging import init_logging, log_evaluation, log_training
from utils.utils import DoneTransform, load_checkpoint, save_checkpoint, save_rollout

SOFT_MODULE_NUM_LAYERS = 2
SOFT_MODULE_NUM_MODULES = 2
SOFT_MODULE_GATING_LAYERS = 2


def _make_linear(in_features: int, out_features: int, device=None) -> nn.Linear:
    if device is None:
        return nn.Linear(in_features, out_features)
    return nn.Linear(in_features, out_features, device=device)


def infer_self_observation_group_slices(
    vmas_env: VmasEnv, agent_index: int = 0
) -> dict[str, slice]:
    agent_index = min(max(agent_index, 0), vmas_env.n_agents - 1)
    td = vmas_env.reset()
    scenario = vmas_env.scenario
    _, obs_self_groups = scenario.observe_self(agent_index, return_groups=True)
    total_obs_dim = td["agents", "observation"].shape[-1]

    group_slices = {}
    cursor = 0
    for name, tensor in obs_self_groups:
        next_cursor = cursor + int(tensor.shape[-1])
        group_slices[name] = slice(cursor, next_cursor)
        cursor = next_cursor

    if cursor > total_obs_dim:
        raise RuntimeError(
            f"Self observation groups exceed total observation dim: {cursor} > {total_obs_dim}."
        )
    return group_slices


class _SoftModularAgentMLP(nn.Module):
    def __init__(
        self,
        *,
        base_input_dim: int,
        condition_dim: int,
        out_features: int,
        device=None,
        depth: int | None = None,
        num_cells: int | list[int] | tuple[int, ...] | None = None,
        activation_class: type[nn.Module] = nn.Tanh,
        num_layers: int = SOFT_MODULE_NUM_LAYERS,
        num_modules: int = SOFT_MODULE_NUM_MODULES,
        module_hidden: int | None = None,
        gating_hidden: int | None = None,
        num_gating_layers: int = SOFT_MODULE_GATING_LAYERS,
        pre_softmax: bool = False,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Soft modular network requires num_layers >= 2.")

        depth = 2 if depth is None else depth
        num_cells = 128 if num_cells is None else num_cells
        hidden_size = num_cells[-1] if isinstance(num_cells, (list, tuple)) else num_cells

        self.activation = activation_class()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.pre_softmax = pre_softmax
        self.module_hidden = hidden_size if module_hidden is None else module_hidden
        self.gating_hidden = self.module_hidden if gating_hidden is None else gating_hidden

        self.obs_encoder = MLP(
            in_features=base_input_dim,
            out_features=hidden_size,
            depth=depth,
            num_cells=num_cells,
            activation_class=activation_class,
            activate_last_layer=False,
            device=device,
        )
        self.condition_encoder = MLP(
            in_features=condition_dim,
            out_features=hidden_size,
            depth=max(depth - 1, 0),
            num_cells=hidden_size,
            activation_class=activation_class,
            activate_last_layer=False,
            device=device,
        )

        if num_gating_layers > 0:
            self.gating_mlp = MLP(
                in_features=hidden_size,
                out_features=self.gating_hidden,
                depth=max(num_gating_layers - 1, 0),
                num_cells=self.gating_hidden,
                activation_class=activation_class,
                activate_last_layer=False,
                device=device,
            )
        else:
            self.gating_mlp = nn.Identity()
            self.gating_hidden = hidden_size

        self.module_layers = nn.ModuleList()
        module_input_dim = hidden_size
        for _ in range(self.num_layers):
            layer_modules = nn.ModuleList(
                [
                    _make_linear(module_input_dim, self.module_hidden, device=device)
                    for _ in range(self.num_modules)
                ]
            )
            self.module_layers.append(layer_modules)
            module_input_dim = self.module_hidden

        self.transition_weight_head0 = _make_linear(
            self.gating_hidden, self.num_modules * self.num_modules, device=device
        )
        self.transition_condition_layers = nn.ModuleList()
        self.transition_weight_heads = nn.ModuleList()
        for layer_idx in range(self.num_layers - 2):
            self.transition_condition_layers.append(
                _make_linear(
                    (layer_idx + 1) * self.num_modules * self.num_modules,
                    self.gating_hidden,
                    device=device,
                )
            )
            self.transition_weight_heads.append(
                _make_linear(
                    self.gating_hidden,
                    self.num_modules * self.num_modules,
                    device=device,
                )
            )
 
        self.final_condition_layer = _make_linear(
            (self.num_layers - 1) * self.num_modules * self.num_modules,
            self.gating_hidden,
            device=device,
        )
        self.final_weight_head = _make_linear(
            self.gating_hidden, self.num_modules, device=device
        )
        self.output_layer = _make_linear(self.module_hidden, out_features, device=device)

    def forward(self, base_inputs: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        obs_features = self.obs_encoder(base_inputs)
        condition_features = self.condition_encoder(condition)
        gating_seed = self.activation(obs_features * condition_features)
        gating_seed = self.gating_mlp(gating_seed)

        weight_shape = (*gating_seed.shape[:-1], self.num_modules, self.num_modules)
        flat_shape = (*gating_seed.shape[:-1], self.num_modules * self.num_modules)

        transition_weights = []
        flattened_transition_weights = []

        raw_weight = self.transition_weight_head0(self.activation(gating_seed)).view(weight_shape)
        soft_weight = torch.softmax(raw_weight, dim=-1)
        transition_weights.append(soft_weight)
        flattened_transition_weights.append(
            raw_weight.view(flat_shape) if self.pre_softmax else soft_weight.view(flat_shape)
        )

        for condition_layer, weight_head in zip(
            self.transition_condition_layers, self.transition_weight_heads
        ):
            transition_context = torch.cat(flattened_transition_weights, dim=-1)
            if self.pre_softmax:
                transition_context = self.activation(transition_context)
            transition_context = condition_layer(transition_context)
            transition_context = self.activation(transition_context * gating_seed)

            raw_weight = weight_head(transition_context).view(weight_shape)
            soft_weight = torch.softmax(raw_weight, dim=-1)
            transition_weights.append(soft_weight)
            flattened_transition_weights.append(
                raw_weight.view(flat_shape)
                if self.pre_softmax
                else soft_weight.view(flat_shape)
            )

        final_context = torch.cat(flattened_transition_weights, dim=-1)
        if self.pre_softmax:
            final_context = self.activation(final_context)
        final_context = self.final_condition_layer(final_context)
        final_context = self.activation(final_context * gating_seed)
        final_weights = torch.softmax(self.final_weight_head(final_context), dim=-1)

        module_outputs = torch.stack(
            [module(obs_features) for module in self.module_layers[0]], dim=-2
        )
        for layer_idx, layer_modules in enumerate(self.module_layers[1:]):
            layer_weight = transition_weights[layer_idx]
            next_outputs = []
            for module_idx, module in enumerate(layer_modules):
                module_input = (
                    module_outputs * layer_weight[..., module_idx, :].unsqueeze(-1)
                ).sum(dim=-2)
                next_outputs.append(module(self.activation(module_input)))
            module_outputs = torch.stack(next_outputs, dim=-2)

        output = (module_outputs * final_weights.unsqueeze(-1)).sum(dim=-2)
        output = self.activation(output)
        return self.output_layer(output)


class SoftModularMultiAgentMLP(nn.Module):
    def __init__(
        self,
        n_agent_inputs: int,
        n_agent_outputs: int,
        n_agents: int,
        *,
        condition_slice: slice,
        centralized: bool | None = None,
        share_params: bool | None = None,
        device=None,
        depth: int | None = None,
        num_cells: int | list[int] | tuple[int, ...] | None = None,
        activation_class: type[nn.Module] = nn.Tanh,
        num_layers: int = SOFT_MODULE_NUM_LAYERS,
        num_modules: int = SOFT_MODULE_NUM_MODULES,
        module_hidden: int | None = None,
        gating_hidden: int | None = None,
        num_gating_layers: int = SOFT_MODULE_GATING_LAYERS,
        pre_softmax: bool = False,
        exclude_condition_from_base: bool = True,
        **kwargs,
    ):
        super().__init__()
        centralized = kwargs.pop("centralised", centralized)
        if centralized is None:
            raise TypeError("centralized arg must be passed.")
        if share_params is None:
            raise TypeError("share_params arg must be passed.")
        if condition_slice is None:
            raise TypeError("condition_slice must be passed.")
        if n_agent_inputs is None:
            raise TypeError("n_agent_inputs must be passed.")

        self.n_agents = n_agents
        self.n_agent_inputs = n_agent_inputs
        self.n_agent_outputs = n_agent_outputs
        self.centralized = centralized
        self.share_params = share_params
        self.condition_slice = condition_slice
        self.exclude_condition_from_base = exclude_condition_from_base
        self.condition_dim = condition_slice.stop - condition_slice.start
        if self.condition_dim <= 0:
            raise ValueError(f"Invalid condition_slice {condition_slice}.")

        self.base_obs_dim = (
            n_agent_inputs - self.condition_dim if exclude_condition_from_base else n_agent_inputs
        )
        if self.base_obs_dim <= 0:
            raise ValueError(
                f"Base observation dim must be positive, got {self.base_obs_dim}."
            )

        base_input_dim = (
            self.base_obs_dim * n_agents if self.centralized else self.base_obs_dim
        )
        agent_network_count = 1 if self.share_params else self.n_agents
        self.agent_networks = nn.ModuleList(
            [
                _SoftModularAgentMLP(
                    base_input_dim=base_input_dim,
                    condition_dim=self.condition_dim,
                    out_features=n_agent_outputs,
                    device=device,
                    depth=depth,
                    num_cells=num_cells,
                    activation_class=activation_class,
                    num_layers=num_layers,
                    num_modules=num_modules,
                    module_hidden=module_hidden,
                    gating_hidden=gating_hidden,
                    num_gating_layers=num_gating_layers,
                    pre_softmax=pre_softmax,
                )
                for _ in range(agent_network_count)
            ]
        )

    def _split_inputs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = obs[..., self.condition_slice]
        if not self.exclude_condition_from_base:
            return obs, condition
        base_obs = torch.cat(
            (obs[..., : self.condition_slice.start], obs[..., self.condition_slice.stop :]),
            dim=-1,
        )
        return base_obs, condition

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-2] != self.n_agents:
            raise ValueError(
                f"Expected obs.shape[-2] == {self.n_agents}, got {obs.shape}."
            )

        base_obs, condition = self._split_inputs(obs)
        if self.centralized:
            shared_base_inputs = base_obs.flatten(-2, -1)
        else:
            shared_base_inputs = base_obs

        if self.share_params:
            if self.centralized:
                base_inputs = shared_base_inputs.unsqueeze(-2).expand(
                    *obs.shape[:-2], self.n_agents, shared_base_inputs.shape[-1]
                )
            else:
                base_inputs = shared_base_inputs
            return self.agent_networks[0](base_inputs, condition)

        outputs = []
        for agent_idx, agent_network in enumerate(self.agent_networks):
            if self.centralized:
                agent_base_inputs = shared_base_inputs
            else:
                agent_base_inputs = shared_base_inputs[..., agent_idx, :]
            agent_condition = condition[..., agent_idx, :]
            outputs.append(agent_network(agent_base_inputs, agent_condition).unsqueeze(-2))
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
    obs_group_slices = infer_self_observation_group_slices(
        base_env, agent_index=min(AGENT_FOCUS_INDEX, base_env.n_agents - 1)
    )
    if "self_hinge_status" not in obs_group_slices:
        raise RuntimeError(
            "self_hinge_status was not found in observe_self() output, "
            "but it is required as the Soft-Module condition input."
        )
    hinge_status_slice = obs_group_slices["self_hinge_status"]
    torchrl_logger.info(
        f"Using self_hinge_status slice [{hinge_status_slice.start}:{hinge_status_slice.stop}] "
        "as the Soft-Module condition input."
    )
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

    actor_net = nn.Sequential(
        SoftModularMultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2
            * env.full_action_spec_unbatched[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=cfg.model.shared_parameters,
            condition_slice=hinge_status_slice,
            device=cfg.train.device,
            depth=2,
            num_cells=128,
            activation_class=nn.Tanh,
            num_layers=SOFT_MODULE_NUM_LAYERS,
            num_modules=SOFT_MODULE_NUM_MODULES,
            num_gating_layers=SOFT_MODULE_GATING_LAYERS,
        ),
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
        centralised=cfg.model.centralised_critic,
        share_params=cfg.model.shared_parameters,
        condition_slice=hinge_status_slice,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
        num_layers=SOFT_MODULE_NUM_LAYERS,
        num_modules=SOFT_MODULE_NUM_MODULES,
        num_gating_layers=SOFT_MODULE_GATING_LAYERS,
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
        entropy_coef=cfg.loss.entropy_eps,
        normalize_advantage=True,
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
        model_name = (
            ("SoftModHet" if not cfg.model.shared_parameters else "SoftMod")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
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
                loss_vals = loss_module(subdata)
                training_tds.append(loss_vals.detach())

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

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
