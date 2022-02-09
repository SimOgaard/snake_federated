# matplotlib imports
from math import fabs
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation

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

class Replay_Snake():

    def __init__(self, board, snake, observation_function: object, test_amount: int = 5, max_step_without_food: int = 100):

        self.board = board
        self.snake = snake
        self.observation_function = observation_function
        self.test_amount = test_amount
        self.max_step_without_food = max_step_without_food

        self.min_val: int = float('inf')
        self.average_val: int = 0
        self.max_val: int = float('-inf')

        self.board.set_snakes(self.snake)

        self.stuck_amount: int = 0
        self.old_board_run: int = self.board.run

    def show_replay_snake(self):
        
        self.min_val: int = float('inf')
        self.average_val: int = 0
        self.max_val: int = float('-inf')

        self.board.set_snakes(self.snake)

        self.stuck_amount: int = 0
        self.old_board_run: int = self.board.run

        self.board.__restart__() # restart board
        self.snake_state = self.observation_function() # save init state

        # store snake lenght and steps with same length
        self.last_snake_length: int = len(self.snake.snake_body)
        self.time_without_food: int = 0

        self.board.run = self.old_board_run

        self.board.__restart__() # restart board
        self.snake_state = self.observation_function() # save init state

        # store snake lenght and steps with same length
        self.last_snake_length: int = len(self.snake.snake_body)
        self.time_without_food: int = 0

        def replay_test_snake(i):

            snake_stuck = False

            action, is_random = self.snake.act(self.snake_state) # choose an action for given snake
            self.snake.move(action, is_random)

            self.snake_state = self.observation_function() # observe what steps taken lead to


            # if snake has been the same length for max_step_without_food steps; break (it got stuck in a loop)
            self.time_without_food += 1
            if (self.last_snake_length != len(self.snake.snake_body)):
                self.last_snake_length = len(self.snake.snake_body)
                self.time_without_food = 0
            elif (self.time_without_food > self.max_step_without_food):
                self.stuck_amount += 1
                snake_stuck = True

            mat.set_data(self.board.board.detach().clone().tolist())

            if not self.board.is_alive() or snake_stuck:
                if (len(self.snake.snake_body) > self.max_val):
                    self.max_val = len(self.snake.snake_body)
                if (len(self.snake.snake_body) < self.min_val):
                    self.min_val = len(self.snake.snake_body)
                self.average_val += len(self.snake.snake_body)
                
                # reset board and snake like this test didnt happen
                self.board.run = self.old_board_run

                self.board.__restart__() # restart board
                self.snake_state = self.observation_function() # save init state

                # store snake lenght and steps with same length
                self.last_snake_length = len(self.snake.snake_body)
                self.time_without_food = 0

                snake_stuck = False

            return mat

        cmap = colors.ListedColormap(['royalblue', 'navy', 'darkred', 'limegreen', 'orange', 'yellow'])

        fig, ax = plt.subplots()

        plt.axis('off')

        mat = ax.matshow([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], cmap=cmap)

        plt.tight_layout()

        print("start animation")

        ani = animation.FuncAnimation(fig, 
        replay_test_snake, 
        interval=10)

        plt.show()