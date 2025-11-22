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
device = torch.device("cpu")

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
        self.M = 4	# number of rules (strategic patterns to learn) - reduced for faster convergence
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

            rule.head = nn.Linear(self.I, self.L)  # For conclusion generation
            
            # LEARNED CONSTANTS: Template values for constant-mode matching
            cs = torch.FloatTensor(self.J, self.L).uniform_(-1,1)
            rule.constants = nn.Parameter(cs)
            
            # CYLINDRIFICATION FACTORS: Constant vs Variable decision
            # γ[j][i] controls whether rule position (j,i) acts as constant or variable
            γs = torch.FloatTensor(self.J, self.L).uniform_(0,1)
            rule.γs = nn.Parameter(γs)
            
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

        FOR EACH RULE:
        1. PATTERN MATCHING: Check how well rule premises match WM
        2. VARIABLE CAPTURE: Extract values into variable slots (when γ≈1)
        3. CONSTANT CHECKING: Verify fixed constraints (when γ≈0)
        4. CONCLUSION GENERATION: Use captured variables to produce output
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
                    γ = self.sigmoid(rule.γs[j, l])
                    constant = rule.constants[j, l]
                    wm_values = state[:, :, l]
                    
                    # Match penalty (lower is better)
                    diff = (constant - wm_values) ** 2
                    match_scores += (1 - γ) * diff
                
                # DIFFERENTIABLE soft attention instead of argmin
                attention_weights = F.softmax(-match_scores / temperature, dim=1)  # [batch, W]
                
                # Soft selection: weighted average instead of hard selection
                best_props = torch.einsum('bw,bwl->bl', attention_weights, state)  # [batch, L]
                
                # Soft match quality
                match_quality += (attention_weights * match_scores).sum(dim=1)
                
                # Capture variables from best match
                captured = rule.body[j](best_props)  # [batch, I]
                
                # Weight by average γ (how much this premise wants to capture)
                γ_avg = self.sigmoid(rule.γs[j, :]).mean()
                captured_vars += γ_avg * captured
            
            # Generate conclusion Q-value from captured variables
            rule_output = rule.head(captured_vars)  # [batch, L]
            
            # Map conclusion to action Q-values
            for w in range(self.W):
                diff = (rule_output - state[:, w, :]) ** 2
                # Use signed similarity so Q can be negative
                similarity = -diff.sum(dim=1)
                
                # Weight by match quality (better match = more confident)
                confidence = torch.exp(-match_quality)
                outputs[:, w] += confidence * similarity
        
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

        q_values = self.anet(state)
        next_q_values = self.tnet(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.anet.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def sync(self):
        self.tnet.load_state_dict(self.anet.state_dict())

    def net_info(self):
        """
        Calculate and return network topology description and total parameters.
        
        Per rule parameters:
        - body networks: J × (L×I + I) weights/biases for premise projections
        - head network: I×L + L weights/biases for conclusion generation  
        - constants: J×L learned template values
        - gammas: J×L cylindrification factors
        
        Total = M × [J×(L×I + I) + (I×L + L) + J×L + J×L]
        """
        M = self.anet.M  # number of rules
        J = self.anet.J  # premises per rule
        I = self.anet.I  # variable slots
        L = self.anet.L  # proposition length
        
        # Per rule calculations
        body_params = J * (L * I + I)  # J linear layers: L→I
        head_params = I * L + L         # 1 linear layer: I→L
        constant_params = J * L         # constants for matching
        gamma_params = J * L            # cylindrification factors
        
        params_per_rule = body_params + head_params + constant_params + gamma_params
        total_params = M * params_per_rule
        
        topology = f"AlgebraicLogic: M={M} rules × [J={J} premises, I={I} vars, L={L} features]"
        
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

