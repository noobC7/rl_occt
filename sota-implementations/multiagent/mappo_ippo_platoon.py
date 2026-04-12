from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

import mappo_ippo_occt as occt_train


def _get_phase_reference(td):
    candidate_keys = (
        ("next", "agents", "reward"),
        ("agents", "reward"),
        ("next", "agents", "done"),
        ("agents", "done"),
        ("agents", "observation"),
        ("next", "agents", "observation"),
    )
    for key in candidate_keys:
        value = occt_train._safe_get(td, key)
        if value is not None:
            return value

    episode_done = occt_train._safe_info_get(td, "episode_done")
    if episode_done is not None:
        return episode_done

    raise KeyError(
        "Could not infer a reference tensor for platoon phase masks."
    )


def extract_training_phase_masks(
    td,
    hinge_key: str = "self_hinge_status",
    agent_hinged_key: str = "agent_hinge_status",
) -> tuple[torch.Tensor, torch.Tensor]:
    del hinge_key, agent_hinged_key
    reference = _get_phase_reference(td)
    mask_shape = (
        reference.shape
        if reference.shape[-1] == 1
        else (*reference.shape[:-1], 1)
    )
    platoon_mask = torch.ones(
        mask_shape,
        device=reference.device,
        dtype=torch.bool,
    )
    hinge_approach_mask = torch.zeros_like(platoon_mask)
    return platoon_mask, hinge_approach_mask


def attach_agent_train_masks(
    td,
    *,
    agent_hinged_key: str = "agent_hinge_status",
):
    del agent_hinged_key
    reward = occt_train._safe_get(td, ("next", "agents", "reward"))
    if reward is None:
        reward = occt_train._safe_get(td, ("agents", "reward"))
    if reward is None:
        reward = _get_phase_reference(td)

    train_active_shape = (
        reward.shape if reward.shape[-1] == 1 else (*reward.shape[:-1], 1)
    )
    train_active = torch.ones(
        train_active_shape,
        device=reward.device,
        dtype=torch.bool,
    )

    env_done = occt_train._safe_get(td, ("next", "agents", "done"))
    if env_done is None:
        env_done = occt_train._safe_get(td, ("next", "done"))
    if env_done is None:
        raise KeyError(
            "Could not find next ('agents', 'done') or env-level ('done') while "
            "building platoon training masks."
        )
    env_done = occt_train._ensure_phase_mask(env_done)
    done_train = occt_train._expand_mask(env_done, train_active)

    td.set(("agents", "train_active"), train_active)
    td.set(("next", "agents", "done_train"), done_train)
    td.set(("next", "agents", "terminated_train"), done_train.clone())
    return td


def extract_agent_train_mask(
    td,
    *,
    train_key: str = "train_active",
    agent_hinged_key: str = "agent_hinge_status",
) -> torch.Tensor:
    del agent_hinged_key
    train_mask = occt_train._safe_get(td, ("agents", train_key))
    if train_mask is not None:
        return occt_train._ensure_phase_mask(train_mask)

    reference = _get_phase_reference(td)
    mask_shape = (
        reference.shape
        if reference.shape[-1] == 1
        else (*reference.shape[:-1], 1)
    )
    return torch.ones(
        mask_shape,
        device=reference.device,
        dtype=torch.bool,
    )


class PlatoonPhaseWeightController:
    def __init__(self, cfg: DictConfig, device: torch.device | str) -> None:
        del cfg
        self.device = torch.device(device)
        self.enabled = False

    def collector_state(self, td) -> dict[str, object]:
        del td
        return {
            "weights": None,
            "metrics": {},
        }


def build_training_summary(
    loss_vals,
    train_active_mask: torch.Tensor,
    platoon_mask: torch.Tensor,
    hinge_approach_mask: torch.Tensor,
    phase_weights,
):
    del platoon_mask, hinge_approach_mask, phase_weights

    summary = occt_train.TensorDict({}, batch_size=[])
    component_totals = []
    loss_device = train_active_mask.device

    for key in ("loss_objective", "loss_critic", "loss_entropy"):
        if key not in loss_vals.keys():
            continue
        component_value = occt_train._masked_tensor_mean(
            loss_vals[key],
            train_active_mask,
        )
        summary.set(key, component_value.detach())
        component_totals.append(component_value)
        loss_device = component_value.device

    if component_totals:
        loss_value = sum(component_totals)
    else:
        loss_value = torch.zeros((), device=loss_device, dtype=torch.float32)

    summary.set("train_active_ratio", train_active_mask.float().mean().detach())

    for metric_key in (
        "entropy",
        "clip_fraction",
        "kl_approx",
        "ESS",
        "explained_variance",
    ):
        metric_value = loss_vals.get(metric_key, None)
        if metric_value is not None:
            summary.set(
                metric_key,
                occt_train._reduce_metric(metric_value).detach(),
            )

    return loss_value, summary


def _patch_occt_training_module() -> None:
    occt_train.extract_training_phase_masks = extract_training_phase_masks
    occt_train.attach_agent_train_masks = attach_agent_train_masks
    occt_train.extract_agent_train_mask = extract_agent_train_mask
    occt_train.MetricAdaptiveWeightController = PlatoonPhaseWeightController
    occt_train.build_training_summary = build_training_summary


@hydra.main(
    version_base="1.1",
    config_path="config/platoon0411",
    config_name="mappo_mlp_baseline",
)
def train(cfg: DictConfig):
    _patch_occt_training_module()
    return occt_train.train.__wrapped__(cfg)


if __name__ == "__main__":
    train()
