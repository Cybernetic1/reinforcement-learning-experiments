python3 -c "
import math

vocab = 30
print(f'Vocabulary size: {vocab}')
print(f'Theoretical minimum bits needed: {math.log2(vocab):.2f}')
print(f'Minimum embedding dim (rounded up): {math.ceil(math.log2(vocab))}')
print()
print('But for neural networks, practical minimums are:')
print(f'  embed_dim=4: Can represent {2**4}=16 orthogonal directions')
print(f'  embed_dim=5: Can represent {2**5}=32 orthogonal directions')
print(f'  embed_dim=6: Can represent {2**6}=64 orthogonal directions')
print()
print('With embed_dim=4:')
print(f'  Embedding table: {vocab} × 4 = {vocab*4} params')
print()
print('Parameter comparison:')
for dim in [2, 3, 4]:
    J, I, L = 2, 3, 4
    embed = vocab * dim
    gammas = J * L
    capture = J * ((L*dim)*(I*dim) + I*dim)
    conclude = (J*I*dim)*(L*vocab) + L*vocab
    per_rule = gammas + capture + conclude
    total = embed + 6*per_rule + 64 + 9
    print(f'  embed_dim={dim:2d}: {per_rule:5,} params/rule → {total:6,} total')
"

python3 -c "
# With optimized settings: M=6, J=2, I=3, L=4, embed_dim=8

J = 2   # premises per rule
I = 3   # variable slots
L = 4   # proposition length
embed_dim = 8
VOCAB_SIZE = 33

print('Per Rule Parameter Breakdown:')
print('=' * 60)

# 1. Gammas (learnable constant/variable switches)
gammas = J * L
print(f'1. Gammas: {J} premises × {L} positions = {gammas}')

# 2. Variable capture layers (one per premise)
# Each captures from L*embed_dim input → I*embed_dim output
capture_per_premise = (L * embed_dim) * (I * embed_dim) + (I * embed_dim)  # weight + bias
capture_total = J * capture_per_premise
print(f'2. Variable Capture ({J} layers):')
print(f'   Per premise: Linear({L}×{embed_dim}={L*embed_dim} → {I}×{embed_dim}={I*embed_dim})')
print(f'   Weight: {L*embed_dim} × {I*embed_dim} = {(L*embed_dim)*(I*embed_dim):,}')
print(f'   Bias: {I*embed_dim}')
print(f'   Per premise total: {capture_per_premise:,}')
print(f'   All {J} premises: {capture_total:,}')

# 3. Conclusion generation layer
# Input: J*I*embed_dim (all captured variables concatenated)
# Output: L*VOCAB_SIZE (logits for each position's symbol)
conclude_input = J * I * embed_dim
conclude_output = L * VOCAB_SIZE
conclude_weight = conclude_input * conclude_output
conclude_bias = conclude_output
conclude_total = conclude_weight + conclude_bias
print(f'3. Conclusion Generator:')
print(f'   Linear({conclude_input} → {conclude_output})')
print(f'   Weight: {conclude_input} × {conclude_output} = {conclude_weight:,}')
print(f'   Bias: {conclude_output}')
print(f'   Total: {conclude_total:,}')

total_per_rule = gammas + capture_total + conclude_total
print(f'\nTOTAL PER RULE: {total_per_rule:,}')

print()
print('The problem is the Conclusion Generator!')
print(f'It outputs {L}×{VOCAB_SIZE} = {L*VOCAB_SIZE} logits to predict')
print(f'{L} symbols (predicate + 3 args) from vocab of {VOCAB_SIZE}')
print(f'This single layer has {conclude_weight:,} parameters!')
"
