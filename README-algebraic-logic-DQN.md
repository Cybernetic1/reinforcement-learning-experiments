# Algebraic Logic Deep Q-Network (DQN)

## Overview

This document explains the **Algebraic Logic Network** architecture in `DQN_logic_with_vars.py` - a novel approach that combines symbolic logic reasoning with deep reinforcement learning for playing Tic-Tac-Toe.

## Core Innovation

Traditional neural networks are "black boxes" - we cannot understand how they make decisions. The Algebraic Logic Network replaces traditional multi-layer perceptrons with **learnable logical rules** that are:
- **Interpretable**: We can inspect what patterns each rule has learned
- **Differentiable**: Rules can be trained using gradient descent
- **Compositional**: Rules can be combined to form complex reasoning

## Architecture Overview

### Input Representation

The Tic-Tac-Toe board is encoded as **9 propositions**, one per square:

```
Board:          Encoding:
X | O | _       [-1, -1.0]  [1, -0.75]  [0, -0.5]
---------   →   [-1, -0.25] [0,  0.0]   [0,  0.25]
_ | _ | _       [0,  0.5]   [0,  0.75]  [0,  1.0]
```

Each square is encoded as `[player, position]`:
- **player**: -1 (opponent), 0 (empty), +1 (self)
- **position**: Normalized spatial location from -1 (top-left) to +1 (bottom-right)

This encoding preserves spatial relationships, allowing rules to learn concepts like "corner" vs "center".

### Logical Rules

The network consists of **M=4 learnable rules**. Each rule has the form:

```
IF (premise_1 matches state) AND (premise_2 matches state)
THEN output Q-values for actions
```

**Key Parameters per Rule:**

1. **rule.constants** [J×L]: Template patterns to match against
   - Example: `[-0.9, -0.8]` means "look for opponent near top-left corner"

2. **rule.γs** (gamma) [J×L]: Constant vs Variable selector
   - γ ≈ 0: Constant mode (match specific pattern)
   - γ ≈ 1: Variable mode (capture any value)

3. **rule.body** [J × Linear(L→I)]: Variable extraction networks
   - Captures features from matched propositions

4. **rule.head** [Linear(I→W)]: Action recommendation network
   - Outputs Q-values for all 9 squares

### Forward Pass (Reasoning Process)

```
1. Input: Board state [batch, 18] 
   ↓ reshape
2. Working Memory: [batch, 9, 2] (9 propositions)
   ↓
3. For each rule:
   a. Match premises against propositions
      - Compute match_scores = (rule.constants - proposition)²
      - Weight by (1 - γ): constants matter when γ≈0
   
   b. Soft attention (CRITICAL for gradients!)
      - attention_weights = softmax(-match_scores / temperature)
      - This is DIFFERENTIABLE (unlike argmin)
   
   c. Variable capture
      - best_props = weighted average of propositions
      - captured = rule.body(best_props)
   
   d. Generate Q-values
      - rule_q_values = rule.head(captured_vars)
   
4. Aggregate: Sum all rule outputs
5. Output: Q-values for 9 actions [batch, 9]
```

## Hybrid Learning Mechanism

The network uses **TWO types of learning simultaneously**:

### 1. Reinforcement Learning (Temporal Credit Assignment)

**What it does:** Determines WHICH actions are good

**Mechanism:** Temporal Difference (TD) Error
```python
TD_error = reward + γ * max Q(next_state, a') - Q(state, action)
```

**Example:**
```
Episode plays out:
t=0: State S0, play action A0 → reward=0
t=1: State S1, play action A1 → reward=0  
t=2: State S2, play action A2 → reward=+20 (WIN!)

RL learns: Actions A0, A1, A2 all contributed to winning
- A2 gets most credit (immediate)
- A1 gets less credit (one step back)
- A0 gets least credit (two steps back)
```

### 2. Gradient-Based Optimization (Structural Credit Assignment)

**What it does:** Determines HOW to adjust rule parameters to favor good actions

**Mechanism:** Backpropagation through entire computation graph
```python
loss = (Q_predicted - Q_target)²
loss.backward()  # Computes ∂loss/∂(every parameter)
optimizer.step() # Updates all parameters
```

**Example:**
```
Suppose action A4 (center square) was good:

loss.backward() computes:
- ∂loss/∂rule.head → "adjust weights to increase Q[4]"
- ∂loss/∂rule.body → "adjust to capture useful features"  
- ∂loss/∂rule.constants → "adjust template to match this pattern"
- ∂loss/∂rule.γs → "adjust to use constant/variable mode better"

All rules are updated to make Q[4] higher for this state!
```

### Why Both Are Necessary

| Component | What It Learns | How |
|-----------|---------------|-----|
| **RL** | Which state-action pairs lead to rewards | TD-error across time |
| **Backprop** | How to adjust network to favor good actions | Chain rule through network |

**Without RL:** Network wouldn't know which actions are good (no reward signal)
**Without Backprop:** Network couldn't adjust internal rule parameters (no gradient flow)

## The Non-Stationarity Challenge

**Problem:** Because rule parameters change every update, the Q-function changes over time:

```
Episode 100: Rules produce Q(state, action) = 2.5
              ↓ (training updates rules)
Episode 200: Same state, but Q(state, action) = 1.8

Q-learning must "chase" a moving target!
```

**Why This Matters:**
- Traditional RL assumes the environment is stationary (fixed transition dynamics)
- Here, the "environment" (how rules compute Q-values) keeps changing
- This can cause instability and prevent convergence

**Solutions Implemented:**

1. **Target Network** (`tnet`):
   - Frozen copy of main network
   - Updated every 100 episodes
   - Provides stable TD targets

2. **Gradient Clipping**:
   - Limits how fast parameters can change
   - Prevents wild oscillations

3. **Experience Replay**:
   - Trains on past experiences
   - Smooths out non-stationarity

4. **Epsilon Decay**:
   - Reduces exploration over time
   - Allows rules to stabilize

## Key Differentiability: Soft Attention

**Critical Design Choice:**

```python
# BAD: Non-differentiable (blocks gradients)
best_idx = argmin(match_scores)
best_prop = state[best_idx]

# GOOD: Differentiable (gradients flow)
attention = softmax(-match_scores / temperature)
best_prop = weighted_sum(attention, state)
```

**Why This Matters:**
- `argmin` selects ONE proposition → gradient only flows to that one
- `softmax` weights ALL propositions → gradients flow to all (proportional to attention)
- This allows rule.constants and rule.γs to receive learning signals!

## Training Process

### Hyperparameters
- Learning rate: 0.001
- Batch size: 256
- Gamma (discount): 0.9
- Epsilon: 1.0 → 0.01 (decays by 0.995 per episode)
- Target sync: Every 100 episodes
- Number of rules: M=4
- Premises per rule: J=2
- Variable slots: I=3

### Training Loop
```python
for episode in episodes:
    state = reset_game()
    
    # Play game
    while not done:
        action = choose_action(state, epsilon)  # ε-greedy
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
    
    # Learn from experience
    if len(replay_buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)
        loss = update(batch)  # RL + Backprop happens here!
    
    # Decay exploration
    epsilon = max(epsilon_min, epsilon * 0.995)
    
    # Sync target network periodically
    if episode % 100 == 0:
        target_net.copy_from(action_net)
```

## Interpretability: Inspecting Learned Rules

Unlike black-box neural networks, we can inspect what each rule has learned:

```python
RL.display_rules()
```

**Example Output:**
```
*** RULE 1 ***
IF:
  Premise 1: Player=Opp (γ=0.12), Pos=top-left (γ=0.24)
  Premise 2: Player=?X (γ=0.85), Pos≈center (γ=0.45)
THEN output Q-values for squares:
  Top 3 preferences:
    Square 4 (center): bias=1.52
    Square 0 (top-left): bias=0.83
    Square 8 (bottom-right): bias=0.21

Interpretation: 
"If opponent at top-left corner, prefer playing center"
```

**What We Learn:**
- Rule 1 has learned to recognize "opponent at corner"
- It recommends playing center (classic Tic-Tac-Toe strategy!)
- γ values show it uses constants for opponent detection (γ≈0) and variables for position capture (γ≈1)

## Advantages Over Standard DQN

| Aspect | Standard DQN | Algebraic Logic DQN |
|--------|-------------|-------------------|
| **Interpretability** | Black box | Can inspect learned rules |
| **Sample Efficiency** | Moderate | Potentially better (structured reasoning) |
| **Generalization** | Weak | Better (rules are compositional) |
| **Debugging** | Hard | Easier (can see which rules fire) |
| **Parameters** | ~3,000-10,000 | ~200-500 (more efficient) |

## Challenges & Future Work

### Current Limitations
1. **Convergence Speed**: Slower than standard DQN due to complexity
2. **Non-Stationarity**: Moving target problem requires careful tuning
3. **Scale**: Tested only on Tic-Tac-Toe (9 actions)

### Potential Improvements
1. **Two-Timescale Learning**: Separate learning rates for rules vs Q-values
2. **Policy Gradient**: Switch from Q-learning to policy gradient for better gradient flow
3. **Working Memory Loop**: Feed rule conclusions back as input for multi-step reasoning
4. **Hierarchical Rules**: Meta-rules that select which object-level rules to activate

### Path to AGI
The algebraic logic approach addresses key requirements for artificial general intelligence:

- **Compositionality**: Rules can be combined to form complex reasoning
- **Transfer Learning**: Learned rules could transfer across tasks
- **Explainability**: Critical for safety and trust
- **Continual Learning**: Can add new rules without forgetting old ones

## Related Work

- **Neural Module Networks** (Andreas et al.): Compositional reasoning
- **Differentiable Neural Computers** (Graves et al.): External memory with attention
- **Logic Tensor Networks** (Serafini & Garcez): First-order logic with neural grounding
- **AlphaGo/MuZero** (DeepMind): Combines search with learned value functions
- **Neuro-Symbolic AI**: Broader field combining symbolic and neural approaches

## References

1. Mnih et al. (2015): "Human-level control through deep reinforcement learning" - Original DQN paper
2. Van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-learning" - Addresses overestimation
3. Schaul et al. (2016): "Prioritized Experience Replay" - Improves sample efficiency
4. Borkar & Meyn (2000): "Two-timescale stochastic approximation" - Theory for non-stationary learning

## Code Structure

```
DQN_logic_with_vars.py
├── ReplayBuffer: Experience replay for off-policy learning
├── AlgelogicNetwork: Core logic network
│   ├── __init__: Initialize M rules with parameters
│   ├── forward: Compute Q-values via rule matching
│   ├── choose_action: ε-greedy action selection
│   └── display_rules: Inspect learned rules
└── DQN: Complete RL agent
    ├── __init__: Initialize networks and optimizer
    ├── update: Hybrid RL+backprop learning
    ├── sync: Copy action net → target net
    └── save_net/load_net: Model persistence
```

## Usage Example

```python
# Initialize agent
from DQN_logic_with_vars import DQN
agent = DQN(action_dim=9, state_dim=18, learning_rate=0.001)

# Training
for episode in range(10000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        
        if len(agent.replay_buffer) > 256:
            agent.update(batch_size=256)
    
    agent.decay_epsilon()
    
    if episode % 100 == 0:
        agent.sync()
        print(f"Episode {episode}, Reward: {reward}")

# Inspect learned strategy
agent.display_rules()

# Save model
agent.save_net("tictactoe_logic_agent")
```

## Conclusion

The Algebraic Logic DQN represents a step toward more interpretable and compositional AI. By replacing black-box neural networks with learnable logical rules, we can:

1. **Understand** what the AI has learned
2. **Debug** when it fails  
3. **Trust** its decisions in critical applications
4. **Extend** to more complex reasoning tasks

While still experimental, this approach shows promise for building AI systems that reason more like humans - using explicit rules that can be inspected, modified, and composed into more complex behaviors.

---

**Author Notes:** This architecture is part of ongoing research into neuro-symbolic AI and differentiable logic. The code is experimental and serves as a proof-of-concept for integrating logical reasoning with deep reinforcement learning.
