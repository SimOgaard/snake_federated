# matplotlib imports
from matplotlib import pyplot as plt

def pretty_print(state, board_dim) -> None:
    '''
    Prints given state in 2d
    '''
    print(chr(27) + "[2J")
    print(state.detach().clone().resize_(board_dim[0], board_dim[1]))

def display(state, board_dim: int) -> None:
    '''
    Plots given state
    '''
    # np.uint8()
    # im = state.detach().clone().resize_(board_dim + 2, board_dim + 2)
    # img = ax.imshow(im, cmap=plt.cm.binary, vmin=0, vmax=10)
    # plt.draw()

from snake_env.snake_agents.observation_functions import observation_full
def display_run(board, snake, board_dim, display_function: object, observation_function: object):
    '''
    Displays a run from given snake
    '''
    board.__restart__() # restart board
    snake_state = observation_function(board = board, snake = snake, kernel = board_dim) # save init state
    visual_state = observation_full(board = board)

    while board.is_alive(): # check if snakes are alive

        display_function(visual_state, board.board.shape)
        input()

        action, is_random = snake.act(snake_state) # choose an action for given snake
        reward: float = snake.move(action, is_random)

        snake_state = observation_function(board = board, snake = snake, kernel = board_dim) # observe what steps taken lead to
        visual_state = observation_full(board = board)
    input("snake died with final length of {}...".format(len(snake.snake_body)))