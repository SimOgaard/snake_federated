# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

# matplotlib imports
import matplotlib.pyplot as plt

def pretty_print(state: FloatTensor) -> None:
    '''
    Prints given state in 2d
    '''
    print(chr(27) + "[2J")
    print(state.resize_(board_dim + 2, board_dim + 2))

def display(state: FloatTensor) -> None:
    '''
    Plots given state
    '''
    plt.imshow(state.resize_(board_dim + 2, board_dim + 2))
    plt.show(block=False)
    plt.pause(1/30)

if __name__ == "__main__":
    '''
    Creates a large environment and counts achived apm of random agent
    '''
    board_dim: int = 9

    player_snake: ControllableAgent = ControllableAgent()
    random_snake: RandomAgent = RandomAgent()

    board: Board = Board(
        min_board_shape         = array([9, 9]),
        max_board_shape         = array([9, 9]),
        salt_and_pepper_chance  = 0.0,
        food_amount             = array([1, 1]),
        replay_interval         = 0,
        snakes                  = [player_snake, random_snake]
    )

    while True:
        board.__restart__()

        while board.is_alive():
            state = observation_full(board = board)
            display(state)
            if (not player_snake.done):
                action: int = player_snake.act()
                reward: float = player_snake.move(action)
            if (not random_snake.done):
                action: int = random_snake.act()
                reward: float = random_snake.move(action)

        input("Every snake is dead...")