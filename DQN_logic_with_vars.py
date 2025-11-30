"""
DQN (Deep Q Network) using my invention "algebraic logic network" to implement Q.

CORE INNOVATION: Instead of traditional neural networks, this uses logical rules
with learnable parameters to represent the Q-function for TicTacToe.

ARCHITECTURE OVERVIEW:
- State representation: 9 propositions (one per board square), each as (player, position) pairs
- Logic rules: M=16 learnable rules of the form "premise → conclusion"
- Each rule has variables that get unified/matched against the working memory (game state)
- Output: Probability distribution over actions (board positions)

BOARD REPRESENTATION (Uniform Encoding):
- Each square encoded as [player, normalized_position]
- player ∈ {-1, 0, 1} for {opponent, empty, self}
- position ∈ [-1, +1] using formula (square_index - 4)/4
- Examples:
  * Square 0 (top-left): [player, -1.0]
  * Square 4 (center): [player, 0.0] 
  * Square 8 (bottom-right): [player, +1.0]
- This preserves spatial relationships crucial for strategic reasoning
- Flattened to [18] array: [player0, pos0, player1, pos1, ..., player8, pos8]

RESEARCH HYPOTHESIS: Logical reasoning with fuzzy unification can learn
interpretable game strategies while maintaining neural network trainability.
The uniform encoding allows rules to naturally learn spatial concepts like
"corner play" (position ≈ ±1) vs "center control" (position ≈ 0).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal

import random
import numpy as np
np.random.seed(7)
torch.manual_seed(7)

# Auto-detect CUDA availability with fallback
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.init()  # Force CUDA initialization
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device} (CUDA not available)")
except Exception as e:
    device = torch.device("cpu")
    print(f"Using device: {device} (CUDA error: {e})")

import types	# for types.SimpleNamespace

class ReplayBuffer:
    """Standard DQN experience replay buffer - unchanged from typical implementations"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def last_reward(self):
        return self.buffer[self.position-1][2]

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = \
            map(np.stack, zip(*batch)) # stack for each element
        '''
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        '''
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class AlgelogicNetwork(nn.Module):
    """
    Algebraic Logic Network - Novel architecture combining symbolic reasoning with neural learning
    
    KEY INNOVATION: Replaces traditional MLP Q-networks with learnable logical rules
    that can perform fuzzy pattern matching and variable unification.
    
    GRADIENT FLOW ARCHITECTURE:
    Input state [batch, 18]
      ↓ reshape
    WM propositions [batch, 9, 2]
      ↓ for each rule:
    Match against rule.constants (adjusted by rule.γs)
      ↓ softmax (differentiable!)
    Soft attention weights
      ↓ einsum
    Selected propositions
      ↓ rule.body (Linear layer)
    Captured variables [batch, I]
      ↓ rule.head (Linear layer)
    Q-values [batch, 9]
    
    ALL STEPS ARE DIFFERENTIABLE - gradients flow from loss back to all rule parameters!
    
    INTERPRETABILITY:
    Unlike black-box MLPs, learned rules can be inspected:
    - rule.constants shows what patterns the rule looks for
    - rule.γs shows which features are treated as constants vs variables
    - rule.head shows which actions the rule recommends
    
    CORE CONCEPTS:
    - Rules are parameterized "if-then" statements with fuzzy matching
    - Variables in rules get "unified" with game state through learned similarity
    - Truth values propagate from premises to conclusions via differentiable logic
    - Each rule learns to recognize strategic patterns (e.g., "block opponent win")
    
    TICTACTOE INTERPRETATION:
    - Input: Current board state as propositions
    - Rules: Strategic patterns like "if corner occupied, play center"
    - Output: Action probabilities based on pattern matching strength
    """

    def __init__(self, input_dim, action_dim, activation=F.relu, learning_rate=3e-3):
        super(AlgelogicNetwork, self).__init__()

        # LOGICAL ARCHITECTURE HYPERPARAMETERS
        # These define the "reasoning capacity" of the network
        self.M = 8	# number of rules (strategic patterns to learn) - increased to 8 for more capacity
        self.J = 2	# number of propositions per rule premise
        self.I = 3	# number of variable slots per rule (for capture/binding)
        self.L = 2	# length of each proposition vector (player + position for TicTacToe)
        self.W = 9	# number of propositions in Working Memory (= board squares)

        # RULE STRUCTURE: Each rule represents a learned strategic insight
        # Format: "IF pattern X matches state THEN conclude action Y with confidence Z"
        self.rules = nn.ModuleList()
        for m in range(0, self.M):
            rule = nn.Module()
            
            # RULE BODY: Maps captured variables to output conclusions
            # "Given what we captured in variables, what should we conclude?"
            rule.body = nn.ModuleList()  # For premise matching/variable capture
            for j in range(self.J):
                rule.body.append(nn.Linear(self.L, self.I))

            rule.head = nn.Linear(self.I, self.W)  # Output Q-values for all 9 squares directly
            
            # LEARNED CONSTANTS: Template values for constant-mode matching
            # Initialized uniformly in [-1,1] to match normalized position encoding
            # These will be adjusted by gradients to recognize spatial patterns
            cs = torch.FloatTensor(self.J, self.L).uniform_(-1,1)
            rule.constants = nn.Parameter(cs)
            
            # CYLINDRIFICATION FACTORS: Constant vs Variable decision
            # γ[j][l] ∈ [0,1] controls whether rule position (j,l) acts as constant or variable
            # γ≈0: constant mode (match specific value)
            # γ≈1: variable mode (capture any value)
            # Initialized uniformly to let network learn which mode works best
            γs = torch.FloatTensor(self.J, self.L).uniform_(0,1)
            rule.γs = nn.Parameter(γs)
            
            # SLOT SELECTORS: Layers that decide which variable slot to use
            # σ[j] maps proposition to slot assignments
            rule.slot_selector = nn.ModuleList()
            for j_idx in range(self.J):
                rule.slot_selector.append(nn.Linear(self.L, self.L * self.I))
            
            self.rules.append(rule)

        self.activation = F.relu

    @staticmethod
    def selector(p, γ):
        """
        ADAPTIVE SELECTOR: Adjust probability based on learned confidence
        
        PURPOSE: Allow rules to learn when to be "strict" vs "flexible" in matching
        - If γ≈0: Output approaches 1 (always fire regardless of match quality)  
        - If γ≈1: Output equals p (fire proportional to match quality)
        
        FORMULA: Homotopy interpolation between certainty and uncertainty
        """
        t = 1.0/(1.0 + torch.exp(-50*(γ - 0.5)))
        return p*t + 1.0 - t

    @staticmethod
    def softmax(x):
        """Temperature-controlled softmax for sharper action selection"""
        β = 5		# temperature parameter (higher = sharper distribution)
        maxes = torch.max(x, 0, keepdim=True)[0]
        x_exp = torch.exp(β * (x - maxes))
        x_exp_sum = torch.sum(x_exp, 0, keepdim=True)
        probs = x_exp / x_exp_sum
        return probs

    @staticmethod
    def sigmoid(γ):
        """Direct clamp to [0,1] - prevents gradient saturation while preserving cylindrification semantics"""
        return torch.clamp(γ, 0.0, 1.0)

    """Let's focus on matching one propositional term in a rule against 
    a working memory (WM) proposition.  Each proposition consists of L=2
    tokens or "symbols" or "positions".  Each position is either a constant
    or a variable, as in first-order logic.  If the position is a constant,
    it is simply compared against the WM content.  If the position is a variable,
    the corresponding WM position is copied into a "variable slot".
    What determines whether a position is constant or variable is the value γ
    ∈ [0, 1], which I termed the "cylindrification factor". """

    @staticmethod
    def match(γ, rule_constant, wm_token):
        """
        CONSTANT/VARIABLE MATCHING
        
        INPUTS:
        - γ: Cylindrification factor (0=constant, 1=variable)
        - rule_constant: Template value for constant-mode matching
        - wm_value: Actual content from Working Memory
        
        BEHAVIOR:
        - γ≈0: Returns match_quality = similarity(rule_constant, wm_value)
        - γ≈1: Returns "perfect match" (allows variable capture)
        
        OUTPUT: Match degree (lower = better match for gradient descent)
        """
        # When γ≈1 (variable mode): match penalty approaches 0 (perfect match)
        # When γ≈0 (constant mode): match penalty = actual difference
        # match_degree = AlgelogicNetwork.sigmoid(γ) * (rule_constant - wm_value)**2
        # return match_degree

    def forward(self, state):
        """
        MAIN MATCHING / UNIFICATION ALGORITHM
    
        GRADIENT FLOW PATH (backward during loss.backward()):
        Q-values → rule_q_values → captured_vars → best_props → match_scores → rule parameters
    
        Each component is differentiable:
        - Soft attention (softmax) allows gradients to flow to match_scores
        - match_scores depends on rule.constants and rule.γs
        - captured_vars depends on rule.body weights
        - rule_q_values depends on rule.head weights
    
        FOR EACH RULE:
        1. PATTERN MATCHING: Check how well rule premises match WM
           → Gradients adjust rule.constants and rule.γs
        2. VARIABLE CAPTURE: Extract values into variable slots (when γ≈1)
           → Gradients adjust rule.body weights
        3. CONCLUSION GENERATION: Use captured variables to produce output
           → Gradients adjust rule.head weights
        """
        batch_size = state.shape[0]
        state = state.reshape(batch_size, self.W, self.L)
        
        # Initialize Q-values (can be negative)
        outputs = torch.zeros(batch_size, self.W)
        
        # Temperature for soft attention (lower = sharper, higher = softer)
        # Higher temp = softer attention = better gradient flow early in training
        temperature = 1.0

        for m in range(self.M):
            rule = self.rules[m]

            # Accumulate captured variables across all premises
            captured_vars = torch.zeros(batch_size, self.I)
            match_quality = torch.zeros(batch_size)

            for j in range(self.J):
                # For each premise, find best matching WM proposition
                match_scores = torch.zeros(batch_size, self.W)

                for l in range(self.L):
                    # LEARNABLE PARAMETERS: rule.γs and rule.constants
                    # Gradients will flow here from final loss
                    γ = self.sigmoid(rule.γs[j, l])
                    constant = rule.constants[j, l]
                    wm_values = state[:, :, l]

                    # Match penalty (lower is better)
                    # When γ→0 (constant mode): penalty = (constant - wm_value)²
                    # When γ→1 (variable mode): penalty → 0 (perfect match)
                    diff = (constant - wm_values) ** 2
                    match_scores += (1 - γ) * diff

                # CRITICAL: DIFFERENTIABLE soft attention instead of argmin
                # argmin blocks gradients, softmax allows gradient flow!
                # Gradients flow: loss → outputs → rule_q_values → captured_vars → best_props → attention_weights → match_scores
                attention_weights = F.softmax(-match_scores / temperature, dim=1)  # [batch, W]

                # Soft selection: weighted average instead of hard selection
                # This allows gradients to flow back to ALL propositions (not just argmin)
                best_props = torch.einsum('bw,bwl->bl', attention_weights, state)  # [batch, L]

                # Soft match quality
                match_quality += (attention_weights * match_scores).sum(dim=1)

                # LEARNABLE PARAMETERS: rule.body[j] weights
                # Capture variables from best match
                captured = rule.body[j](best_props)  # [batch, I]

                # Weight by average γ (how much this premise wants to capture)
                γ_avg = self.sigmoid(rule.γs[j, :]).mean()
                captured_vars += γ_avg * captured

                # For each premise, decide which slot gets which proposition element
                # Example: If premise has [player, position], we need to decide:
                #   - Does 'player' value go to slot 0, 1, or 2?
                #   - Does 'position' value go to slot 0, 1, or 2?

                # Use softmax to make this decision:
                slot_logits = rule.slot_selector[j](best_props)  # [batch, L * I]
                slot_probs = F.softmax(slot_logits.view(batch_size, self.L, self.I), dim=2)  # [batch, L, I]

                # Each of L=2 positions picks ONE slot via softmax
                for l in range(self.L):
                    slot_weights = slot_probs[:, l, :]  # [batch, I] - distribution over I slots
                    captured_vars += slot_weights * best_props[:, l].unsqueeze(1)  # Soft assignment

            # Generate Q-values directly for all 9 squares from captured variables
            rule_q_values = rule.head(captured_vars)  # [batch, W=9]
            
            # Weight by match quality (better match = more confident)
            confidence = torch.exp(-match_quality).unsqueeze(1)  # [batch, 1]
            outputs += confidence * rule_q_values  # [batch, W]
        
        return outputs

    def choose_action(self, state, epsilon=0.1, deterministic=False):
        """
        ACTION SELECTION for TicTacToe using logical reasoning
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self(state_tensor).squeeze(0)  # [9]
        
        # Mask invalid actions
        state_reshaped = state_tensor.reshape(self.W, self.L)
        valid_mask = (state_reshaped[:, 0] == 0)  # Empty squares
        
        if not valid_mask.any():
            return random.randint(0, 8)  # Fallback if board full
        
        # Set invalid actions to -inf
        q_values[~valid_mask] = float('-inf')
        
        # Epsilon-greedy exploration during training
        if not deterministic and random.random() < epsilon:
            valid_actions = torch.where(valid_mask)[0]
            return valid_actions[random.randint(0, len(valid_actions)-1)].item()
        
        return torch.argmax(q_values).item()


class DQN():
    """
    Deep Q-Network with Algebraic Logic Network architecture
    
    HYBRID LEARNING APPROACH:
    This combines two types of learning:
    
    1. REINFORCEMENT LEARNING (Temporal Credit Assignment):
       - TD-error (reward + γ*max Q(s',a') - Q(s,a)) tells us WHICH actions are good
       - Assigns credit across TIME: action at t=0 leads to reward at t=5
       - Implemented in update() via expected_q_value vs q_value
    
    2. GRADIENT-BASED OPTIMIZATION (Structural Credit Assignment):
       - Backpropagation tells us HOW to adjust rule parameters to favor good actions
       - Assigns credit across NETWORK: input → rules → Q-values
       - Implemented in update() via loss.backward()
    
    NON-STATIONARITY CHALLENGE:
    Since rule parameters change every update (via optimizer.step()), the Q-function
    changes over time. This creates a "moving target" problem:
    - Episode 100: Rules produce Q(s,a) = 2.5
    - Episode 200: Same state, but rules now produce Q(s,a) = 1.8
    
    MITIGATION STRATEGIES:
    - Target network (tnet): Provides stable TD targets, synced every 100 episodes
    - Gradient clipping: Limits how fast rules can change
    - Experience replay: Averages over past experiences
    - Epsilon decay: Gradually reduces exploration as rules stabilize
    
    If training is unstable, consider:
    - Slower learning rate for rule parameters (two-timescale optimization)
    - More frequent target network syncing
    - Soft target updates (Polyak averaging)
    """
    def __init__(self, action_dim, state_dim, learning_rate=3e-4, gamma=0.9):
        self.anet = AlgelogicNetwork(state_dim, action_dim, learning_rate=learning_rate).to(device)
        self.tnet = AlgelogicNetwork(state_dim, action_dim, learning_rate=learning_rate).to(device)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.gamma = gamma
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.anet.parameters(), lr=learning_rate)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Define terminal/end state with properly normalized position encoding
        # Each square: [player=0, normalized_position]
        # Position formula: (square_index - 4) / 4 maps [0,8] to [-1,+1]
        self.endState = [
            0, -1.0,    # Square 0: (0-4)/4 = -1.0
            0, -0.75,   # Square 1: (1-4)/4 = -0.75
            0, -0.5,    # Square 2: (2-4)/4 = -0.5
            0, -0.25,   # Square 3: (3-4)/4 = -0.25
            0,  0.0,    # Square 4: (4-4)/4 = 0.0
            0,  0.25,   # Square 5: (5-4)/4 = 0.25
            0,  0.5,    # Square 6: (6-4)/4 = 0.5
            0,  0.75,   # Square 7: (7-4)/4 = 0.75
            0,  1.0     # Square 8: (8-4)/4 = 1.0
        ]

    def choose_action(self, state, deterministic=False):
        return self.anet.choose_action(
            state, 
            epsilon=self.epsilon if not deterministic else 0.0,
            deterministic=deterministic
        )
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update(self, batch_size, reward_scale=1.0):
        if len(self.replay_buffer) < batch_size:
            return None
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        # FORWARD PASS: Compute Q-values through algebraic logic network
        # This builds a computation graph: state → [rule matching] → [variable capture] → Q-values
        # All rule parameters (constants, γs, body, head) are part of this graph
        q_values = self.anet(state)
        next_q_values = self.tnet(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        
        # TEMPORAL DIFFERENCE ERROR: RL component
        # This is where RL provides the learning signal - which actions led to good rewards
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        # BACKPROPAGATION: Gradient flow through logic rules
        # This single backward() call computes gradients for ALL rule parameters:
        # - ∂loss/∂rule.constants (how to adjust pattern templates)
        # - ∂loss/∂rule.γs (how to adjust constant/variable behavior)
        # - ∂loss/∂rule.body (how to capture variables better)
        # - ∂loss/∂rule.head (how to generate better Q-values from variables)
        #
        # KEY INSIGHT: This is HYBRID learning:
        # - RL (TD-error) tells us WHICH actions are good (temporal credit assignment)
        # - Backprop tells us HOW to adjust rules to favor those actions (structural credit assignment)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping prevents rule parameters from changing too fast
        # This helps maintain stability in the non-stationary learning process
        torch.nn.utils.clip_grad_norm_(self.anet.parameters(), 1.0)
        
        # UPDATE ALL PARAMETERS: Rules learn to produce better Q-values
        # After this step, the SAME state will produce DIFFERENT Q-values
        # This creates non-stationarity - the "moving target" problem
        self.optimizer.step()
        
        return loss.item()

    def sync(self):
        self.tnet.load_state_dict(self.anet.state_dict())

    def save_net(self, filename):
        """Save the action network (anet) parameters to file"""
        import os
        os.makedirs("PyTorch_models", exist_ok=True)
        filepath = f"PyTorch_models/{filename}.dict"
        torch.save(self.anet.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_net(self, filename):
        """Load the action network (anet) parameters from file and sync to target network"""
        filepath = f"PyTorch_models/{filename}.dict"
        self.anet.load_state_dict(torch.load(filepath))
        self.sync()  # Also update target network
        print(f"Model loaded from {filepath}")

    def display_rules(self):
        """Display the learned logical rules in human-readable format"""
        print("\n" + "="*80)
        print("LEARNED LOGICAL RULES")
        print("="*80)

        def interpret_player(val):
            """Interpret player value"""
            if val < -0.5: return "Opp"
            elif val > 0.5: return "Self"
            else: return "Empty"

        def interpret_position(val):
            """Interpret normalized position [-1, +1]"""
            if val < -0.6: return "top-left"
            elif val < -0.2: return "top/left"
            elif val < 0.2: return "center"
            elif val < 0.6: return "bottom/right"
            else: return "bottom-right"
        
        # Get a sample empty board state to compute Q-values
        empty_state = torch.FloatTensor(self.endState).unsqueeze(0)
        
        for m, rule in enumerate(self.anet.rules):
            print(f"\n*** RULE {m+1} ***")
            
            # Display premises with cylindrification factors
            print("IF:")
            for j in range(self.anet.J):
                γ_vals = rule.γs[j, :].detach()
                const_vals = rule.constants[j, :].detach()

                γ_player = γ_vals[0].item()
                γ_position = γ_vals[1].item()
                const_player = const_vals[0].item()
                const_position = constVals[1].item()

                print(f"  Premise {j+1}: ", end="")

                # Player matching
                if γ_player < 0.3:
                    print(f"Player={interpret_player(const_player)}", end="")
                elif γ_player > 0.7:
                    print(f"Player=?X", end="")
                else:
                    print(f"Player≈{interpret_player(const_player)}", end="")
                
                print(f" (γ={γ_player:.2f}), ", end="")
                
                # Position matching
                if γ_position < 0.3:
                    print(f"Pos={interpret_position(const_position)}", end="")
                elif γ_position > 0.7:
                    print(f"Pos=?Y", end="")
                else:
                    print(f"Pos≈{interpret_position(const_position)}", end="")
                
                print(f" (γ={γ_position:.2f})")

            # Display conclusion - direct Q-values for 9 squares
            print("THEN output Q-values for squares:")
            head_weights = rule.head.weight.detach()  # [W, I]
            head_bias = rule.head.bias.detach()  # [W]
            
            # Show which squares this rule prefers (based on bias)
            sorted_squares = torch.argsort(head_bias, descending=True)
            print(f"  Top 3 square preferences (by bias):")
            for i in range(min(3, self.anet.W)):
                sq = sorted_squares[i].item()
                pos = (sq - 4) / 4
                print(f"    Square {sq} ({interpret_position(pos)}): bias={head_bias[sq].item():.2f}")
            
            print(f"  Weight matrix norm: {head_weights.norm().item():.3f}")

        print("\n" + "="*80)
        print(f"Total Rules: {self.anet.M}")
        print("NOTE: Each rule matches premises against board state, captures variables,")
        print("      and directly outputs Q-values for all 9 squares.")
        print("="*80 + "\n")

    def net_info(self):
        """
        Calculate and return network topology description and total parameters.
        
        Per rule parameters:
        - body networks: J × (L×I + I) weights/biases for premise projections
        - head network: I×W + W weights/biases for Q-value generation (all 9 squares)
        - constants: J×L learned template values
        - gammas: J×L cylindrification factors
        
        Total = M × [J×(L×I + I) + (I×W + W) + J×L + J×L]
        """
        M = self.anet.M  # number of rules
        J = self.anet.J  # premises per rule
        I = self.anet.I  # variable slots
        L = self.anet.L  # proposition length
        W = self.anet.W  # number of squares
        
        # Per rule calculations
        body_params = J * (L * I + I)  # J linear layers: L→I
        head_params = I * W + W         # 1 linear layer: I→W (direct Q-values)
        constant_params = J * L         # constants for matching
        gamma_params = J * L            # cylindrification factors
        
        params_per_rule = body_params + head_params + constant_params + gamma_params
        total_params = M * params_per_rule
        
        topology = f"AlgebraicLogic.M={M}x(J={J},I={I},L={L})"
        return (topology, total_params)

    def play_random(self, state, action_space):
        """
        Random baseline player that only selects valid (empty) squares.
        
        For the algebraic logic network with uniform encoding:
        - state format: [player0, pos0, player1, pos1, ..., player8, pos8]
        - player = 0 indicates empty square
        """
        # Find all empty squares (where player value is 0)
        empty_squares = []
        for i in range(9):
            player_idx = i * 2  # player is at even indices
            if state[player_idx] == 0:
                empty_squares.append(i)
        
        # If no empty squares (shouldn't happen in valid game), return random
        if not empty_squares:
            return random.randint(0, 8)
        
        # Select random empty square
        return random.choice(empty_squares)

