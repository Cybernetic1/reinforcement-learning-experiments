from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='TicTacToe-v1',
    entry_point='gym_tictactoe.tic_tac_toe:TicTacToeEnv',
    reward_threshold=1000
)