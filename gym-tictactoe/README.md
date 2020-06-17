# Gym TicTacToe
---------
Gym TicTacToe is a light Tic-Tac-Toe environment for OpenAI Gym.

## Installation
1. Install [OpenAi Gym](https://github.com/openai/gym)
```bash
pip install gym
```

2. Download and install `gym-tictactoe`
```bash
git clone https://github.com/ClementRomac/gym-tictactoe
cd gym-tictactoe
python setup.py install
```

## Running
Start by importing the package and initializing the environment
```python
import gym
import gym_tictactoe
env = gym.make('TicTacToe-v1', symbols=[-1, 1], board_size=3, win_size=3) 
```

As the TicTacToe is a two players game, you have to create two players (here we use random as action choosing strategy). The environment is not handling the two players part, so you have to do it in your code as shown below.
```python
user = 0
done = False
reward = 0

# Reset the env before playing
state = env.reset()

while not done:
    env.render(mode=None)
    if user == 0:
        state, reward, done, infos = env.step(env.action_space.sample(), -1)
    elif user == 1:
        state, reward, done, infos = env.step(env.action_space.sample(), 1)
       
    # If the game isn't over, change the current player
    if not done:
        user = 0 if user == 1 else 1
    else :
        if reward == 10:
            print("Draw !")
        elif reward == -20:
            print("Infos : " + str(infos))
            if user == 0:
                print("Random wins ! AI Reward : " + str(reward))
            elif user == 1:
                print("AI wins ! AI Reward : " + str(-reward))
        elif reward == 20:
            if user == 0:
                print("AI wins ! AI Reward : " + str(reward))
            elif user == 1:
                print("Random wins ! AI Reward : " + str(reward))
```

*Warning : If you play on a position where you or your opponent already played, you'll get a 'bad_position' reward and will loose the game*

## Settings
You can change the rewards by editing the `settings.xml` placed in your `gym-tictactoe` installation folder..
