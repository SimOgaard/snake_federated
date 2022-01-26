# matplotlib imports
from math import fabs
from matplotlib import pyplot as plt

def pretty_print(state, board_dim) -> None:
    '''
    Prints given state in 2d.
    For larger board sizes use display function
    '''
    print(chr(27) + "[2J")
    print(state.detach().clone().resize_(board_dim[0], board_dim[1]))

def display(state, board_dim: int) -> None:
    '''
    Plots given state
    Is yet to be implemented
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
    snake_state = observation_function() # save init state
    visual_state = observation_full(board = board)

    while board.is_alive(): # check if snakes are alive
        display_function(visual_state, board_dim)

        action, is_random = snake.act(snake_state) # choose an action for given snake
        reward: float = snake.move(action, is_random)

        snake_state = observation_function() # observe what steps taken lead to
        visual_state = observation_full(board = board)
    input("snake died with final length of {}...".format(len(snake.snake_body)))

from snake_replay import Replay

def test_snake(board, snake, observation_function: object, test_amount: int = 5, print_every: int = 5, max_step_without_food: int = 2_500, visualize: bool = False):
    '''
    Tests snake multiple times and saves min, average, max
    '''    
    min_val: int = float('inf')
    average_val: int = 0
    max_val: int = float('-inf')

    board.set_snakes(snake)

    stuck_amount: int = 0
    old_board_run: int = board.run

    #if (visualize):
    #    replay: Replay = Replay()
    #    replay.replay_snake()

    for i in range(test_amount):
        board.__restart__() # restart board
        snake_state = observation_function() # save init state

        # store snake lenght and steps with same length
        last_snake_length: int = len(snake.snake_body)
        time_without_food: int = 0

        while board.is_alive(): # check if snakes are alive
            action, is_random = snake.act(snake_state) # choose an action for given snake
            snake.move(action, is_random)

            snake_state = observation_function() # observe what steps taken lead to

            if (visualize):
                #replay.append_replay(board.board.detach().clone())
                pretty_print(snake_state, array([7,7]))
                print(snake_state[-4:])
                input(len(snake.snake_body))

            # if snake has been the same length for max_step_without_food steps; break (it got stuck in a loop)
            time_without_food += 1
            if (last_snake_length != len(snake.snake_body)):
                last_snake_length = len(snake.snake_body)
                time_without_food = 0
            elif (time_without_food > max_step_without_food):
                stuck_amount += 1
                break

        if (len(snake.snake_body) > max_val):
            max_val = len(snake.snake_body)
        if (len(snake.snake_body) < min_val):
            min_val = len(snake.snake_body)
        average_val += len(snake.snake_body)

        if ((i + 1) % print_every == 0 and i != 0):
            print("Min: {}; Average: {}; Max: {}; Stuck: {}".format(min_val, average_val/(i + 1), max_val, stuck_amount))
        
        # reset board and snake like this test didnt happen
        board.run = old_board_run