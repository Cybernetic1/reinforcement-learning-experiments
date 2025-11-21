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

    def __init__(self, input_dim, action_dim, hidden_size, activation=F.relu, init_w=3e-3):
        super(AlgelogicNetwork, self).__init__()

        # LOGICAL ARCHITECTURE HYPERPARAMETERS
        # These define the "reasoning capacity" of the network
        self.M = 16	# number of rules (strategic patterns to learn)
        self.J = 2	# number of propositions per rule premise
        self.I = 3	# number of variable slots per rule (for capture/binding)
        self.L = 2	# length of each proposition vector (player + position for TicTacToe)
        self.W = 9	# number of propositions in Working Memory (= board squares)

        # RULE STRUCTURE: Each rule represents a learned strategic insight
        # Format: "IF pattern X matches state THEN conclude action Y with confidence Z"
        self.rules = nn.ModuleList()
        for m in range(0, self.M):
            rule = nn.Module()
            
            # RULE TAIL: Maps captured variables to output conclusions
            # "Given what we captured in variables, what should we conclude?"
            rule.tail = nn.Linear(self.I, self.L)
            
            # RULE HEAD: Neural networks for variable capture/projection
            # "How to copy/transform WM content into variable slots"
            rule.head = nn.ModuleList()
            for j in range(0, self.J):
                rule.head.append(nn.Linear(self.L, self.I))
            
            # LEARNED CONSTANTS: Template values for constant-mode matching
            cs = torch.FloatTensor(self.J + 1, self.L).uniform_(-1,1)
            rule.constants = nn.Parameter(cs)
            
            # CYLINDRIFICATION FACTORS: Constant vs Variable decision
            # γ[j][i] controls whether rule position (j,i) acts as constant or variable
            γs = torch.FloatTensor(self.J + 1, self.L).uniform_(0,1)
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
        """Steep sigmoid for crisp fuzzy logic operations"""
        steepness = 10.0
        t = 1.0/(1.0 + torch.exp(-steepness*(γ - 0.5)))
        return t

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
        state = state.reshape(batch_size, self.W, self.L)  # [batch, 9 squares, 2 features]
        
        # Output will be [batch_size, W] representing Q-values for each action/square
        outputs = torch.zeros(batch_size, self.W)
        
        for m in range(self.M):  # For each rule
            rule = self.rules[m]
            
            # STEP 1: MATCH EACH PREMISE AGAINST ALL WM PROPOSITIONS
            # Track which WM props satisfy each premise
            premise_matches = []  # Will hold [J, batch, W] match scores
            premise_captures = []  # Will hold [J, batch, W, I] captured variables
            
            for j in range(self.J):  # For each premise in the rule
                # Match this premise against all WM propositions
                match_scores = torch.zeros(batch_size, self.W)
                captures = torch.zeros(batch_size, self.W, self.I)
                
                for l in range(self.L):  # For each position (player, location)
                    γ = torch.sigmoid(rule.γs[j, l])  # 0=constant, 1=variable
                    constant = rule.constants[j, l]
                    wm_values = state[:, :, l]  # [batch, W]
                    
                    # Compute match quality (lower is better)
                    diff = (constant - wm_values) ** 2
                    match_penalty = (1 - γ) * diff  # Only penalize if constant mode
                    match_scores += match_penalty
                    
                    # Capture variables (project WM content into variable slots)
                    # When γ≈1, this captures; when γ≈0, captures nothing
                    projection = rule.head[j](state[:, :, l:l+1])  # [batch, W, I]
                    captures += γ * projection.squeeze(-1)
                
                premise_matches.append(match_scores)  # [batch, W]
                premise_captures.append(captures)      # [batch, W, I]
            
            # STEP 2: AGGREGATE MATCHES ACROSS PREMISES
            # For each WM prop, sum match quality across all premises
            premise_matches = torch.stack(premise_matches)  # [J, batch, W]
            total_match = premise_matches.sum(dim=0)  # [batch, W]
            
            # STEP 3: SELECT BEST MATCHING WM PROPOSITIONS
            # For each batch, find the W proposition that best satisfies all premises
            best_indices = torch.argmin(total_match, dim=1)  # [batch]
            
            # Gather the captured variables from best matching props
            premise_captures = torch.stack(premise_captures)  # [J, batch, W, I]
            captured_vars = torch.zeros(batch_size, self.I)
            for b in range(batch_size):
                for j in range(self.J):
                    captured_vars[b] += premise_captures[j, b, best_indices[b], :]
        
            # STEP 4: GENERATE CONCLUSION FROM CAPTURED VARIABLES
            # The tail network maps captured variables to output space
            conclusion = rule.tail(captured_vars)  # [batch, L]
            
            # STEP 5: MATCH CONCLUSION AGAINST ACTION SPACE
            # The conclusion should match one of the WM propositions (squares)
            # to indicate which action to take
            action_scores = torch.zeros(batch_size, self.W)
            for w in range(self.W):
                # How well does the conclusion match this action/square?
                diff = (conclusion - state[:, w, :]) ** 2
                similarity = torch.exp(-diff.sum(dim=1))  # Higher = better match
                action_scores[:, w] += similarity
            
            # Weight by rule's overall match quality
            rule_confidence = torch.exp(-total_match.min(dim=1).values)  # [batch]
            outputs += action_scores * rule_confidence.unsqueeze(1)
        
        return outputs  # [batch_size, 9] Q-values for each square
"""
BOARD REPRESENTATION (Updated):
- Each square encoded as [player, normalized_position]  
- player ∈ {-1, 0, 1} for {opponent, empty, self}
- position ∈ [-1, +1] where (square_index - 4)/4
- Center square → [?, 0.0], Corners → [?, ±1.0, ±0.75]
- This preserves spatial relationships crucial for strategic reasoning
"""

def choose_action(self, state, deterministic=True):
    """
    ACTION SELECTION for TicTacToe using logical reasoning
    
    INPUT: state = [18] array: [player0, pos0, player1, pos1, ..., player8, pos8]
           - player ∈ {-1,0,1} for opponent/empty/self
           - pos ∈ [-1,+1] normalized position encoding
           
    OUTPUT: action ∈ {0,1,2,3,4,5,6,7,8} representing board square
    
    STRATEGIC ADVANTAGE: Normalized encoding allows rules to learn
    spatial concepts like "corner play" or "center control" naturally
    """
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    logits = self.anet(state)  # Get action probabilities from logic rules
    probs  = torch.softmax(logits, dim=1)
    dist   = Categorical(probs)
    action = dist.sample().numpy()[0]
    return action

    def update(self, batch_size, reward_scale, gamma=0.99):
        """
        Q-LEARNING UPDATE using logical rule conclusions
        
        INNOVATION: Instead of updating MLP weights, this updates:
        1. Rule pattern templates (constants)
        2. Matching fuzziness parameters (gammas)  
        3. Variable unification mappings (head/tail networks)
        
        GOAL: Rules learn to recognize winning/losing patterns and
        adjust their strategic recommendations accordingly
        """
        alpha = 1.0  # trade-off between exploration (max entropy) and exploitation (max Q)

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.LongTensor(action).to(device)
        reward     = torch.FloatTensor(reward).to(device)
        done       = torch.BoolTensor(done).to(device)

        # Get Q-values from logical reasoning network
        logits = self.anet(state)
        next_logits = self.anet(next_state)

        # BELLMAN EQUATION with logical Q-function
        # Q_logic(s,a) += η[R + γ max_a' Q_logic(s',a') - Q_logic(s,a)]
        # This trains the logical rules to predict long-term strategic value
        q = logits[range(logits.shape[0]), action]
        m = torch.max(next_logits, 1, keepdim=False).values
        target_q = torch.where(done, reward, reward + self.gamma * m)
        q_loss = self.q_criterion(q, target_q.detach())

        # GRADIENT UPDATE: Adjust logical reasoning parameters
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return

    def net_info(self):
        """Network architecture description"""
        config = "(9)-9-9-(9)"
        neurons = config.split('-')
        last_n = 9
        total = 0
        for n in neurons[1:-1]:
            n = int(n)
            total += last_n * n
            last_n = n
        total += last_n * 9
        return (config, total)

    def play_random(self, state, action_space):
        """
        RANDOM BASELINE PLAYER for TicTacToe
        
        LOGIC: Only select from available (empty) squares
        This provides a sanity check for the logical reasoner
        """
        empties = [0,1,2,3,4,5,6,7,8]
        
        # Find occupied squares by scanning state propositions
        for i in range(0, 18, 2):  # Step by 2 for (symbol, position) pairs
            proposition = state[i : i + 2]
            sym = proposition[0]
            if sym == 1 or sym == -1:  # If square is occupied
                x = proposition[1]
                j = x + 4  # Convert position encoding to square index  
                empties.remove(j)
                
        # Randomly select from available squares
        action = random.sample(empties, 1)[0]
        return action

    def save_net(self, fname):
        """TODO: Implement saving of logical rule parameters"""
        print("Model not saved.")

    def load_net(self, fname):
        """TODO: Implement loading of logical rule parameters"""  
        print("Model not loaded.")
