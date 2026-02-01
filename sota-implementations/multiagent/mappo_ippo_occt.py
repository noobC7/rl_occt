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
def rendering_callback(env, td):
    env.frames.append(env.render(mode="rgb_array", agent_index_focus=round(env.scenario.n_agents/2)-1)) 

def rendering_batch_callback(env, td):
    for env_index in range(env.num_envs):
        env.frames[env_index].append(env.render(mode="rgb_array", agent_index_focus=round(env.scenario.n_agents/2)-1, env_index=env_index)) 

@hydra.main(version_base="1.1", config_path="config", config_name="mappo_ippo_platoon")
def train(cfg: DictConfig):
    # Device
    cfg.train.device = "cpu" if not torch.cuda.device_count() else "cuda:0"
    cfg.env.device = cfg.train.device
    cfg.env.max_steps = eval(cfg.env.max_steps)
    torch.manual_seed(cfg.seed)
    resume_from_checkpoint = cfg.train.resume_from_checkpoint
    start_iteration = 0
    start_frames = 0
    # Sampling
    cfg.collector.total_frames = cfg.collector.frames_per_batch * cfg.collector.n_iters
    cfg.buffer.memory_size = cfg.collector.frames_per_batch
    cfg.env.scenario.eval_mode = False
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
    if cfg.collector.n_iters == 0:
        # only evaluation, fix the map path batch id
        cfg.env.scenario.eval_mode = True 
    env_test = VmasEnv(
        scenario=cfg.env.scenario_name,
        num_envs=cfg.eval.evaluation_episodes,
        continuous_actions=True,
        max_steps=cfg.env.eval_max_steps,
        device=cfg.env.device,
        seed=cfg.seed,
        **cfg.env.scenario,
    )

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
            num_cells=256,
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
        start_iteration, start_frames = load_checkpoint(
            resume_from_checkpoint, policy, value_module, optim
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
        model_name = (
            ("Het" if not cfg.model.shared_parameters else "")
            + ("MA" if cfg.model.centralised_critic else "I")
            + "PPO"
        )
        logger = init_logging(cfg, model_name)

    total_time = 0
    total_frames = start_frames
    if cfg.collector.n_iters == 0:
        print(f"[OcctCRMap] Path num: {env_test.scenario.get_occt_cr_path_num()}. Start Evaluation.")
        evaluation_start = time.time()
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            env_test.frames = [[] for _ in range(env_test.num_envs)]
            rollouts = env_test.rollout(
                max_steps=cfg.env.eval_max_steps,
                policy=policy,
                callback=rendering_batch_callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
                break_when_all_done=True,
            )
            save_rollout(logger, rollouts, start_iteration, total_frames)
            evaluation_time = time.time() - evaluation_start
            print(f"[OcctCRMap] Evaluation rollout finished, duration: {evaluation_time:.2f}s.")
            # log_evaluation(logger, rollouts, env_test, evaluation_time, \
            #                 step=start_iteration, video_caption=f"iter_{start_iteration}_path_{0}.mp4")
            log_batch_video(logger, rollouts, env_test, iter = start_iteration)
            video_encode_time = time.time() - evaluation_start - evaluation_time
            print(f"[OcctCRMap] Evaluation video encode finished, duration: {video_encode_time:.2f}s.")

        if not env.is_closed:
            env.close()
        if not env_test.is_closed:
            env_test.close()
        return
    
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
            and i % cfg.eval.evaluation_interval == 0
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
                eval_path_idx = int(env_test.scenario.road.batch_id[0].cpu().numpy())
                evaluation_time = time.time() - evaluation_start
                log_evaluation(logger, rollouts, env_test, evaluation_time, 
                               step=i, video_caption=f"path{eval_path_idx}")
                save_checkpoint(logger, policy, value_module, optim, i, total_frames)
                #save_rollout(logger, rollouts, i, total_frames, suffix=f"_path{eval_path_idx}")

        sampling_start = time.time()
    collector.shutdown()
    if not env.is_closed:
        env.close()
    if not env_test.is_closed:
        env_test.close()
if __name__ == "__main__":
    train()