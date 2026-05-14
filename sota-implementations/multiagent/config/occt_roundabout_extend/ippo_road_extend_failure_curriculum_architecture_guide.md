# IPPO Failure Curriculum Actor-Critic Architecture Guide

## Scope

This document describes the actor-critic network that is actually selected by:

- `sota-implementations/multiagent/config/occt_roundabout_extend/ippo_road_extend_failure_curriculum.yaml`

and:

- `sota-implementations/multiagent/mappo_ippo_occt.py`

The goal of this document is not to explain PPO in general, but to help draw a correct network architecture figure.

## One-Sentence Summary

The current branch uses:

- `IPPO`
- `5` agents
- shared actor parameters across all agents
- shared critic parameters across all agents
- decentralized actor
- decentralized critic
- plain MLP actor
- plain MLP critic
- no LSTM
- no retentive branch
- no LipsNet
- no phase-conditioned / bi-head critic

## Configuration Switches That Determine the Architecture

From `ippo_road_extend_failure_curriculum.yaml`:

```yaml
model:
  shared_parameters: True
  centralised_critic: False
  is_actor_retentive: False
  is_critic_bi_head: False
  actor:
    depth: 2
    num_cells: 192
  critic:
    depth: 2
    num_cells: 688
env:
  scenario:
    use_history_observation: False
    n_agents: 5
```

These switches imply:

- one shared actor MLP is reused by all `5` agents
- one shared critic MLP is reused by all `5` agents
- each agent uses only its own observation for both actor and critic
- only the current-frame observation is used
- failure curriculum affects sampling/reset logic, not the actor/critic topology

## Drawing Principles

When drawing the figure, use these three principles:

1. Draw the architecture as `per-agent local policy/value estimation`, not as a global centralized network.
2. Show that actor and critic are two separate networks, but each one is parameter-shared across agents.
3. Show that the actor MLP does not directly output the final action. It first outputs distribution parameters, then samples an action through `TanhNormal`.

## High-Level Topology

Recommended top-level layout:

```text
Agent observations (5 agents, named observation dict)
        |                         |
        |                         |
        v                         v
   Shared Actor Path         Shared Critic Path
        |                         |
        v                         v
   action distribution       per-agent scalar value
        |                         |
        v                         v
   sampled action a_i            V_i(o_i)
```

Important:

- do not draw a single global observation entering the critic
- do not draw recurrent memory states
- do not draw a phase gate, dual heads, or Lipschitz branch

## Actor Path

### Functional Form

For agent `i`:

```text
o_i
 -> NamedObservationProjector
 -> shared local MLP
 -> raw actor output
 -> NormalParamExtractor
 -> TanhNormal
 -> action a_i
```

### Actor Input

The actor input is the per-agent named observation dict.

Because:

- `use_history_observation: False`
- `actor.fields.include: null`
- `actor.fields.exclude: [self_hinge_status, self_hinge_past_info]`

the actor uses all available current-frame observation fields except:

- `self_hinge_status`
- `self_hinge_past_info`

For drawing, it is better to group the actor input as semantic blocks instead of listing every scalar in order:

- ego state
  - `self_vel`
  - `self_speed`
  - `self_steering`
  - `self_acc`
- reference/path features
  - `self_ref_velocity`
  - `self_ref_points`
- hinge / geometry / boundary features
  - `self_hinge_velocity`
  - `self_hinge_points`
  - `self_left_boundary_distance`
  - `self_right_boundary_distance`
  - `self_distance_to_ref`
  - `self_distance_to_hinge`
  - `self_distance_to_left_boundary`
  - `self_distance_to_right_boundary`
- tracking / platoon error features
  - `self_error_vel`
  - `self_platoon_error_space`
- nearby-agent relation features if exposed by the scenario
  - `others_pos`
  - `others_rot`
  - `others_relative_longitudinal_velocity`
  - `others_relative_acceleration`
  - `others_distance`

Note:

- the exact concatenation order is decided at runtime by the observation key order
- for a figure, grouped labels are clearer than showing the exact flatten order

### NamedObservationProjector

`NamedObservationProjector` does:

- read selected fields from the named observation dict
- keep only the current frame
- flatten each field
- concatenate all selected fields into one vector

So the correct box label is:

```text
NamedObservationProjector
(field select + flatten + concatenate)
```

### Actor MLP Structure

The actor branch uses:

- `depth = 2`
- `num_cells = 192`
- `activation = Tanh`

Therefore the actor MLP should be drawn as:

```text
Linear(d_actor, 192)
-> Tanh
-> Linear(192, 192)
-> Tanh
-> Linear(192, 2 * action_dim)
```

For the OCCT control setting, the action is the 2-D continuous control:

- acceleration
- steering

so:

```text
action_dim = 2
```

and the last actor linear layer outputs:

```text
2 * action_dim = 4
```

Thus the concrete actor output layer can be drawn as:

```text
Linear(192, 4)
```

### Distribution Parameter Extraction

The actor MLP output is not the final action.

Let the last MLP output be:

\[
z = [z_1, z_2, z_3, z_4]
\]

`NormalParamExtractor` splits it into:

\[
\mu = [z_1, z_2]
\]

\[
\sigma =
\left[
\ln(1 + e^{z_3 + 0.5254587}) + 0.01,
\ln(1 + e^{z_4 + 0.5254587}) + 0.01
\right]
\]

So the best drawing is:

```text
raw actor output (4)
-> NormalParamExtractor
-> loc (2) + scale (2)
```

### Action Sampling

The policy distribution is `TanhNormal`, so the action path is:

\[
u \sim \mathcal{N}(\mu, \sigma)
\]

\[
a = \tanh(u)
\]

If the environment action bounds are not exactly `[-1, 1]`, `TanhNormal` also applies the final affine scaling to the actual action range.

For the figure, the recommended label is:

```text
TanhNormal(loc, scale)
-> sampled action a_i in R^2
```

If you want a more control-oriented label, use:

```text
sampled 2-D continuous control
(e.g. acceleration and steering)
```

If the exact environment-side action order is not important for the figure, do not force an order label. Writing `2-D continuous control` is safer.

## Critic Path

### Functional Form

For agent `i`:

```text
o_i
 -> NamedObservationProjector
 -> shared local critic MLP
 -> scalar value V_i(o_i)
```

### Critic Input

Because:

- `centralised_critic: False`
- `critic.fields.include: null`
- `critic.fields.exclude: [self_hinge_status]`

the critic also uses a per-agent current-frame observation dict, and it excludes only:

- `self_hinge_status`

This means:

- the critic is still decentralized
- it does not concatenate all agents' observations
- it is not a MAPPO-style centralized value network in this config

### Critic MLP Structure

The critic branch uses:

- `depth = 2`
- `num_cells = 688`
- `activation = Tanh`

So draw it as:

```text
Linear(d_critic, 688)
-> Tanh
-> Linear(688, 688)
-> Tanh
-> Linear(688, 1)
```

The output is:

```text
per-agent scalar value
V_i(o_i)
```

## Multi-Agent Sharing: How to Draw It Correctly

This is the part most likely to be drawn incorrectly.

### Correct Interpretation

- there are `5` agents
- each agent has its own observation and action
- the actor network is shared across agents
- the critic network is shared across agents
- actor parameters and critic parameters are not shared with each other

Mathematically:

\[
a_i \sim \pi_\theta(\cdot \mid o_i), \quad i = 1,\dots,5
\]

\[
V_i = V_\phi(o_i), \quad i = 1,\dots,5
\]

where:

- `\theta` is one shared actor parameter set
- `\phi` is one shared critic parameter set

### Best Visual Encoding

Use one of the following styles:

1. Draw one actor box and write `shared across 5 agents`.
2. Draw 5 parallel agent lanes, but annotate the actor and critic boxes with `shared weights`.
3. Draw one shared actor module and one shared critic module, each fed by a small `x5 agents` marker.

Avoid drawing:

- 5 different actor networks with different colors
- a global critic that takes `o_1, ..., o_5` concatenated together

## What Should Not Appear in the Figure

The current configuration does not use these modules, so they should not appear:

- LSTM
- temporal memory state
- retentive branches
- self/hinge/others multi-branch fusion
- LipsNet blocks
- Jacobian regularization blocks
- phase-conditioned shared trunk + bi-head output
- centralized critic input
- old observation history stack

Also note:

- failure replay curriculum is a training-data mechanism
- it is not part of the actor or critic forward graph

So if you want to mention it, place it outside the network figure as a training-side annotation only.

## Recommended Final Figure Layout

### Option A: Clean Paper Figure

Top row:

- `Agent i observation dict`
- `NamedObservationProjector`
- `Shared Actor MLP`
- `NormalParamExtractor`
- `TanhNormal`
- `Action a_i`

Bottom row:

- `Agent i observation dict`
- `NamedObservationProjector`
- `Shared Critic MLP`
- `Scalar value V_i`

At the left or top corner, add:

- `5 agents`
- `parameter sharing across agents`
- `IPPO`

### Option B: Multi-Agent Figure

Show:

- `Agent 1 ... Agent 5`
- each agent feeds the same actor module
- each agent feeds the same critic module

Then annotate:

- `local observation`
- `shared weights`
- `decentralized actor`
- `decentralized critic`

## Minimal Text Labels for the Boxes

Use the following labels directly if you want a compact figure:

### Actor

```text
Per-agent observation dict
NamedObservationProjector
(select + flatten + concat)
Shared Actor MLP
Linear(d_actor,192)-Tanh-Linear(192,192)-Tanh-Linear(192,4)
NormalParamExtractor
loc(2), scale(2)
TanhNormal
2-D action a_i
```

### Critic

```text
Per-agent observation dict
NamedObservationProjector
(select + flatten + concat)
Shared Critic MLP
Linear(d_critic,688)-Tanh-Linear(688,688)-Tanh-Linear(688,1)
Scalar value V_i(o_i)
```

## Common Mistakes Checklist

Before finalizing the figure, check:

- actor last layer is `4`, not `2`
- final action dimension is `2`
- `NormalParamExtractor` is shown between actor MLP and action distribution
- critic output is `1`
- critic is drawn as decentralized, not centralized
- actor and critic are shown as two separate parameter sets
- parameter sharing is across agents, not between actor and critic
- failure curriculum is not drawn inside the neural network

## Ready-to-Use Prompt for Diagram Generation

If you want to feed a text-to-diagram model, this prompt is a good starting point:

```text
Draw a clean actor-critic architecture diagram for a 5-agent IPPO continuous-control system.
Use parameter sharing across agents.
The actor is decentralized and receives each agent's own current-frame named observation dictionary.
Show a NamedObservationProjector that selects fields, flattens them, and concatenates them into a per-agent vector.
Then show a shared actor MLP:
Linear(d_actor,192) -> Tanh -> Linear(192,192) -> Tanh -> Linear(192,4).
Then show NormalParamExtractor splitting the 4 outputs into loc(2) and scale(2).
Then show a TanhNormal distribution producing a 2-D action [acceleration, steering].
The critic is also decentralized and parameter-shared across agents.
Show a separate NamedObservationProjector and a shared critic MLP:
Linear(d_critic,688) -> Tanh -> Linear(688,688) -> Tanh -> Linear(688,1),
outputting a scalar value V_i(o_i) for each agent.
Do not draw centralized critic inputs, recurrent modules, LipsNet blocks, retentive branches, or phase-conditioned heads.
Add a small note that failure replay curriculum is a training-data mechanism outside the network graph.
```

## Code Anchors

Useful code anchors if you need to verify the figure later:

- config switches:
  - `sota-implementations/multiagent/config/occt_roundabout_extend/ippo_road_extend_failure_curriculum.yaml`
- observation key resolution:
  - `resolve_model_obs_keys`
- observation projector:
  - `NamedObservationProjector`
- actor construction:
  - `actor_backbone`
  - `NormalParamExtractor`
  - `ProbabilisticActor`
- critic construction:
  - `value_module`
  - `MultiAgentMLP`

## Final Drawing Verdict

If you only remember one template, use this:

```text
For each of 5 agents:
observation dict
 -> projector
 -> shared actor MLP
 -> NormalParamExtractor
 -> TanhNormal
 -> 2-D action

observation dict
 -> projector
 -> shared critic MLP
 -> scalar value
```

That is the correct architecture for the current `ippo_road_extend_failure_curriculum.yaml` branch.
