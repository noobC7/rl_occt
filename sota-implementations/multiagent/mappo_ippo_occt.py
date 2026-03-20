import time
import os
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
from torchrl.modules.models.multiagent import MultiAgentMLP,GroupSharedMLP
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

@hydra.main(version_base="1.1", config_path="config", config_name="mappo_occt_3_followers")
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

    if cfg.model.shared_parameters:
        share_params = torch.tensor([0] * cfg.env.scenario.n_agents , device=cfg.train.device)
    else:
        share_params = torch.tensor([0] + [1] * (cfg.env.scenario.n_agents - 1), device=cfg.train.device)
    # Policy
    actor_net = nn.Sequential(
        GroupSharedMLP( 
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2
            * env.full_action_spec_unbatched[env.action_key].shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_params,
            device=cfg.train.device,
            depth=2,
            num_cells=128,
            activation_class=nn.Tanh,
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

    # Critic
    module = GroupSharedMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=cfg.model.centralised_critic,
        share_params=share_params,
        device=cfg.train.device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
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

    # Loss
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=value_module,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coeff=cfg.loss.entropy_eps,
        normalize_advantage=True,
        normalize_advantage_exclude_dims=(-2,),
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
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
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
