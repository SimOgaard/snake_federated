# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

# matplotlib imports
from matplotlib import pyplot as plt

def pretty_print(state: FloatTensor, board_dim: int) -> None:
    '''
    Prints given state in 2d
    '''
    print(chr(27) + "[2J")
    print(state.resize_(board_dim + 2, board_dim + 2))

def display(state: FloatTensor, board_dim: int) -> None:
    '''
    Plots given state
    '''
    # np.uint8()
    im = state.detach().clone().resize_(board_dim + 2, board_dim + 2)
    img = ax.imshow(im, cmap=plt.cm.binary, vmin=0, vmax=10)
    plt.draw()

if __name__ == "__main__":
    '''
    Creates a environment with controllable agent and a random agent, is playable
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
        snakes                  = [player_snake, random_snake],
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

    while True:
        board.__restart__()

        while board.is_alive():
            state = observation_full(board = board)
            display(state, board_dim)
            if (not player_snake.done):
                action: int = player_snake.act()
                reward: float = player_snake.move(action)
            if (not random_snake.done):
                action: int = random_snake.act()
                reward: float = random_snake.move(action)

        input("Every snake is dead...")