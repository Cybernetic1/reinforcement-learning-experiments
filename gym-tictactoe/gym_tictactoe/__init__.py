from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='TicTacToe-v1',
    entry_point='gym_tictactoe.tic_tac_toe1:TicTacToeEnv',
    reward_threshold=1000
)

register(
    id='TicTacToe-v2',
    entry_point='gym_tictactoe.tic_tac_toe2:TicTacToeEnv',
    reward_threshold=1000
)
