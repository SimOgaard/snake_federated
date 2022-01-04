# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *
from snake_terminal import display, pretty_print
from snake_training import load_checkpoint

# Math modules
from numpy import mean as numpy_mean

# Torch imports
from torch import tensor, save, load

# Generic imports
from os import path

def display_run(board: Board, snake: Snake, board_dim: array):
    board.__restart__() # restart board
    state: FloatTensor = observation_full(board = board) # save init state

    while board.is_alive(): # check if snakes are alive

        pretty_print(state, board_dim)
        input()

        action: int = snake.act(state) # choose an action for given snake
        reward: float = snake.move(action)

        state: FloatTensor = observation_full(board = board) # observe what steps taken lead to
    input("snake died with final length of {}...".format(len(snake.snake_body)))

if __name__ == "__main__":
    '''
    Trains a DQN-agent
    '''

    episode_amount: int = 100_000
    board_dim: int = 5
    state_size: int = 5
    model_id: str = "{}x{}".format(state_size, state_size)
    model_path: str = 'snake_rl/models/checkpoint{}.pth'.format(model_id)

    dqn_snake: DQNAgent = DQNAgent(
        state_size    = state_size**2,
        action_size   = 4,
        init_snake_lengths=array([2, 2]),
        seed          = 1337,
        batch_size    = 128,
        gamma         = 0.999,
        epsilon_start = 0.,
        epsilon_end   = 0.,
        epsilon_decay = 5000,
        learning_rate = 5e-4,
        tau           = 1e-3,
        update_every  = 32,
        buffer_size   = 500_000
    )

    board: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile()
        },
    )

    if (path.exists(model_path)): # load model
        checkpoint = load(model_path)
        load_checkpoint(dqn_snake)

    while board.run < episode_amount:
        display_run(board, dqn_snake, array([board_dim+2, board_dim+2]))