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

def load_checkpoint(checkpoint_path, policy, value_module, optim):
    """加载训练检查点并恢复训练状态"""
    checkpoint = torch.load(checkpoint_path)
    
    policy.load_state_dict(checkpoint['policy_state_dict'])
    value_module.load_state_dict(checkpoint['value_module_state_dict'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        torchrl_logger.warning("Optimizer state dict not found in checkpoint.")
    
    iteration = checkpoint['iteration']
    total_frames = checkpoint['total_frames']
    
    torchrl_logger.info(f"Checkpoint loaded from {checkpoint_path}")
    torchrl_logger.info(f"Resuming training from iteration {iteration}, frame {total_frames}")
    
    return iteration, total_frames