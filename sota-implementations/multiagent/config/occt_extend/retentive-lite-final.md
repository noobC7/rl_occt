# Retentive Lite Final

## Goal

Reduce the current retentive actor capacity down to the same level as the baseline MLP actor, so that:

- training budget can stay unchanged
- comparison is closer to a capacity-matched ablation
- any gain or drop is more likely to come from temporal structure, not from model size inflation

## Baseline Reference

Current baseline config:

- `config/occt_extend/mappo_road_extend_baseline.yaml`

Current baseline actor:

- `depth = 2`
- `num_cells = 192`
- actor parameter count: `50,308`

## Parameter Search Result

I checked a grid of:

- `actor.num_cells`
- `actor.retentive.branch_hidden`
- `actor.retentive.lstm_hidden`

using the current code and current observation field set.

The closest match to the baseline actor parameter count is:

- `actor.num_cells = 96`
- `branch_hidden = 48`
- `lstm_hidden = 24`

Resulting retentive actor parameter count:

- `50,212`

Difference vs baseline actor:

- `-96`

This is effectively parameter-matched.

## Final Lite Config

File:

- `config/occt_extend/mappo_road_extend_is_actor_retentive_lite.yaml`

Final key values:

```yaml
model:
  is_actor_retentive: True
  actor:
    depth: 2
    num_cells: 96
    retentive:
      branch_hidden: 48
      lstm_hidden: 24
```

## Current Field Set

### Current-frame self branch

- `self_vel`
- `self_speed`
- `self_steering`
- `self_acc`
- `self_ref_velocity`
- `self_ref_points`
- `self_hinge_preview_info`
- `self_distance_to_ref`
- `self_left_boundary_distance`
- `self_right_boundary_distance`
- `self_distance_to_left_boundary`
- `self_distance_to_right_boundary`
- `self_platoon_error_vel`
- `self_hinge_error_vel`
- `self_platoon_error_space`

### Hinge temporal branch

- `self_hinge_past_info`

### Other-agent temporal branch

- `others_pos`
- `others_rot`
- `others_relative_longitudinal_velocity`
- `others_relative_acceleration`
- `others_distance`

## Why This Setting Is Reasonable

This choice keeps:

- the retentive structure
- temporal modeling of hinge information
- temporal modeling of other-agent motion
- current-frame road geometry cues

while removing the previous “actor much larger than baseline” problem.

The comparison is now much cleaner:

- baseline actor params: `50,308`
- retentive-lite actor params: `50,212`

So if performance changes, the explanation is much more likely to be:

- temporal inductive bias

rather than:

- capacity mismatch

## Important Note

This is actor-matched, not necessarily branch-by-branch behavior-matched.

That means:

- optimization dynamics can still differ
- retentive actor may still converge differently
- but the “it wins/loses only because it is much bigger” criticism is largely removed

## Suggested Use

Use this config as the main fair retentive-actor comparison against:

- `config/occt_extend/mappo_road_extend_baseline.yaml`

If this version still underperforms, the likely causes are no longer “too many actor parameters”, but instead:

- temporal modeling hurts the control mapping
- current/history feature partition is still suboptimal
- critic/actor information mismatch
- PPO optimization is not benefiting from the recurrent structure
