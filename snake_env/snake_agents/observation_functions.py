# Math modules
from numpy import array

# Torch imports
from torch import FloatTensor, flatten

# Repo imports
from snake_env.snake_agents.virtual_snake import Snake
from snake_env.snake_environment import Board

def observation_full(board: Board) -> FloatTensor:
    '''
    Returns the whole board
    '''
    return flatten(board.board.detach().clone())

def observation_near(board: Board, snake: Snake, snake_head: array, kernel: array) -> FloatTensor:
    '''
    Returns kernel sized state around head
    '''
    point_u = snake_head + snake.all_actions[1]
    point_r = snake_head + snake.all_actions[2]
    point_d = snake_head + snake.all_actions[0]
    point_l = snake_head + snake.all_actions[3]

    try:

        state = [
            # What is up
            board.board[point_u[0]][point_u[1]],
            
            # What is right
            board.board[point_r[0]][point_r[1]],

            # What is down
            board.board[point_d[0]][point_d[1]],

            # What is left
            board.board[point_l[0]][point_l[1]],

            # Current_direction
            snake.action
        ]
        return FloatTensor(state)
    except:
        return FloatTensor([5, 5, 5, 5, snake.action])

def observation_to_bool(tensor: FloatTensor) -> FloatTensor:
    '''
    Returns bool tensor of input tensor
    '''
    return tensor.where(tensor == 0 or tensor == 0, 1, 0.)

def observation_food(snake_head: array, board: Board) -> FloatTensor:
    '''
    Returns bool tensor where nearest food is [up, right, down, left]
    '''
    return FloatTensor([1, 0, 0, 0])
