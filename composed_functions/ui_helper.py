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

from snake_env.snake_agents.observation_functions import *

def display_run(board, snake, board_dim, display_function: object, observation_function: object):
    '''
    Displays a run on board from snake
    '''
    board.__restart__() # restart board
    #snake_state = observation_cat(observation_near(board=board, snake=snake, kernel=board_dim), observation_food(snake)) # save init state
    snake_state = observation_function() # save init state
    visual_state = observation_full(board = board)

    while board.is_alive(): # check if snakes are alive
        display_function(visual_state, board_dim)
        #input()

        action, is_random = snake.act(snake_state) # choose an action for given snake
        reward: float = snake.move(action, is_random)

        snake_state = observation_function() # observe what steps taken lead to
        # snake_state = observation_cat(observation_near(board=board, snake=snake, kernel=board_dim), observation_food(snake)) # save init state
        visual_state = observation_full(board = board)
    input("snake died with final length of {}...".format(len(snake.snake_body)))

def test_snake(board, snake, observation_function: object):
    '''
    Tests snake multiple times and saves min, average, max
    '''    
    min_val: int = float('inf')
    average_val: int = 0
    max_val: int = float('-inf')

    while True:
        board.__restart__() # restart board
        snake_state = observation_function() # save init state

        while board.is_alive(): # check if snakes are alive
            action, is_random = snake.act(snake_state) # choose an action for given snake
            snake.move(action, is_random)

            snake_state = observation_function() # observe what steps taken lead to

        if (len(snake.snake_body) > max_val):
            max_val = len(snake.snake_body)
        if (len(snake.snake_body) < min_val):
            min_val = len(snake.snake_body)
        average_val += len(snake.snake_body)

        print("Min: {}; Average: {}; Max: {}".format(min_val, average_val/(board.run+1), max_val))