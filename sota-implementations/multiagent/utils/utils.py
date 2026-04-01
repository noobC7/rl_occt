# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from tensordict import unravel_key
from torchrl.envs import Transform


def swap_last(source, dest):
    source = unravel_key(source)
    dest = unravel_key(dest)
    if isinstance(source, str):
        if isinstance(dest, str):
            return dest
        return dest[-1]
    if isinstance(dest, str):
        return source[:-1] + (dest,)
    return source[:-1] + (dest[-1],)


class DoneTransform(Transform):
    """Expands the 'done' entries (incl. terminated) to match the reward shape.

    Can be appended to a replay buffer or a collector.
    """

    def __init__(self, reward_key, done_keys):
        super().__init__()
        self.reward_key = reward_key
        self.done_keys = done_keys

    def forward(self, tensordict):
        for done_key in self.done_keys:
            new_name = swap_last(self.reward_key, done_key)
            tensordict.set(
                ("next", new_name),
                tensordict.get(("next", done_key))
                .unsqueeze(-1)
                .expand(tensordict.get(("next", self.reward_key)).shape),
            )
        return tensordict
import torch
import os
from torchrl._utils import logger as torchrl_logger

def save_checkpoint(logger, policy, value_module, optim, iteration, total_frames):
    """保存训练检查点"""
    checkpoint_dir = os.path.join(logger.save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, 
        f"checkpoint_iter_{iteration}_frames_{total_frames}.pt"
    )
    
    checkpoint = {
        'iteration': iteration,
        'total_frames': total_frames,
        'policy_state_dict': policy.state_dict(),
        'value_module_state_dict': value_module.state_dict(),
        'optimizer_state_dict': optim.state_dict()
    }
    
    torch.save(checkpoint, checkpoint_path)
    torchrl_logger.info(f"Checkpoint saved at {checkpoint_path}")
def save_rollout(logger, rollouts, iteration, total_frames, suffix=""):
    """
    保存rollout对象到rollout文件夹中
    
    参数:
        logger: 日志记录器对象，需包含save_dir属性
        rollouts: rollout输出的TensorDict对象
        iteration: 当前迭代次数
        total_frames: 总帧数
    
    返回:
        rollout_path: 保存的rollout文件路径
    """
    # 创建rollout保存目录
    rollout_dir = os.path.join(logger.experiment.public.run_dir,"rollouts")
    os.makedirs(rollout_dir, exist_ok=True)
    
    # 构建保存路径
    rollout_path = os.path.join(
        rollout_dir, 
        f"rollout_iter_{iteration}_frames_{total_frames}{suffix}.pt"
    )
    
    # 保存rollout对象
    torch.save(rollouts, rollout_path)
    print(f"Rollout saved at {rollout_path}")
    
    return rollout_path


def _load_module_checkpoint(
    module,
    state_dict,
    *,
    label: str,
    flexible: bool,
):
    if not flexible:
        module.load_state_dict(state_dict)
        return

    module_state = module.state_dict()
    compatible_state = {}
    skipped_unexpected = []
    skipped_shape = []

    for key, value in state_dict.items():
        if key not in module_state:
            skipped_unexpected.append(key)
            continue
        if module_state[key].shape != value.shape:
            skipped_shape.append(
                (key, tuple(value.shape), tuple(module_state[key].shape))
            )
            continue
        compatible_state[key] = value

    load_result = module.load_state_dict(compatible_state, strict=False)
    if skipped_unexpected or skipped_shape or load_result.missing_keys:
        warning_lines = [
            f"Loaded {len(compatible_state)}/{len(module_state)} compatible tensors into {label}."
        ]
        if skipped_unexpected:
            warning_lines.append(
                f"Skipped {len(skipped_unexpected)} unexpected tensors in checkpoint for {label}."
            )
        if skipped_shape:
            warning_lines.append(
                f"Skipped {len(skipped_shape)} shape-mismatched tensors in {label}."
            )
        if load_result.missing_keys:
            warning_lines.append(
                f"{label} still has {len(load_result.missing_keys)} missing tensors after partial load."
            )
        torchrl_logger.warning(" ".join(warning_lines))


def load_checkpoint(
    checkpoint_path,
    policy=None,
    value_module=None,
    optim=None,
    *,
    flexible: bool = False,
    resume_mode: str = "resume",
):
    """加载训练检查点并恢复训练状态
    resume：完整续训。加载 policy、value_module、optimizer，并且继续使用 checkpoint 里的 iteration 和 total_frames。也就是“从上次断点接着跑”。
    warm_start：权重热启动。加载 policy 和 value_module，但不加载 optimizer，同时把返回的 iteration 和 total_frames 重置为 0。也就是“拿旧模型参数当初始化，但按一次新训练开始”。
    fine_tune：当前仓库里的定义是“只加载 policy，不加载 value_module，也不加载 optimizer”，并把 iteration 和 total_frames 重置为 0。也就是“只迁移 actor，critic 重新学”。
    """
    if resume_mode not in {"resume", "warm_start", "fine_tune"}:
        raise ValueError(f"Unsupported resume_mode: {resume_mode}")

    checkpoint = torch.load(checkpoint_path)
    target_value_module = value_module if resume_mode != "fine_tune" else None
    target_optim = optim if resume_mode == "resume" else None

    if policy is not None and "policy_state_dict" in checkpoint:
        _load_module_checkpoint(
            policy,
            checkpoint["policy_state_dict"],
            label="policy",
            flexible=flexible,
        )
    if target_value_module is not None and "value_module_state_dict" in checkpoint:
        _load_module_checkpoint(
            target_value_module,
            checkpoint["value_module_state_dict"],
            label="value_module",
            flexible=flexible,
        )
    if target_optim is not None and "optimizer_state_dict" in checkpoint:
        if flexible:
            try:
                target_optim.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                torchrl_logger.warning(
                    "Skipped optimizer state restore because the optimizer structure changed."
                )
        else:
            target_optim.load_state_dict(checkpoint["optimizer_state_dict"])

    iteration = checkpoint.get("iteration", 0)
    total_frames = checkpoint.get("total_frames", 0)
    if resume_mode != "resume":
        iteration, total_frames = 0, 0

    action = "partially loaded" if flexible else "loaded"
    torchrl_logger.info(
        f"Checkpoint {action} from {checkpoint_path} with resume_mode={resume_mode}"
    )
    torchrl_logger.info(
        f"Resuming training from iteration {iteration}, frame {total_frames}"
    )

    return iteration, total_frames
