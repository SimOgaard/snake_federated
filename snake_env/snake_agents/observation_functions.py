# Math modules
from numpy import array

# Torch imports
from torch import FloatTensor, flatten
from torch.nn import ConstantPad2d

# Repo imports
from snake_env.snake_agents.virtual_snake import Snake
from snake_env.snake_environment import Board

def observation_full(board: Board) -> FloatTensor:
    '''
    Returns the whole board
    '''
    return flatten(board.board.detach().clone())

def observation_near(board: Board, snake: Snake, kernel: array) -> FloatTensor:
    '''
    Returns kernel sized state around head
    '''
    def clamp_coord(val:int, min_val:int, max_val:int) -> int:
        '''
        Returns clamped val between min_val and max_val
        '''
        if (val < min_val):
            return min_val
        if val > max_val:
            return max_val
        return val

    kernel = (kernel - array([1, 1])) // 2
    snake_head: array = snake.snake_body[0]

    # get x coordinates 
    x_from: int = snake_head[0] - kernel[0]
    x_from_clamped: int = clamp_coord(x_from, 0, board.max_board_shape[0] + 1)
    x_to: int = snake_head[0] + kernel[0] + 1
    x_to_clamped: int = clamp_coord(x_to, 0, board.max_board_shape[0] + 2)

    # get y coordinates 
    y_from: int = snake_head[1] - kernel[1]
    y_from_clamped: int = clamp_coord(y_from, 0, board.max_board_shape[1] + 1)
    y_to: int = snake_head[1] + kernel[1] + 1
    y_to_clamped: int = clamp_coord(y_to, 0, board.max_board_shape[1] + 2)

    clipped_tensor = board.board[x_from_clamped:x_to_clamped,y_from_clamped:y_to_clamped]

    diff_left: int = y_from_clamped - y_from
    padding_left: int = diff_left if (diff_left > 0) else 0
    diff_right: int = y_to - y_to_clamped
    padding_right: int = diff_right if (diff_right > 0) else 0
    diff_up: int = x_from_clamped - x_from
    padding_up: int = diff_up if (diff_up > 0) else 0
    diff_down: int = x_to - x_to_clamped
    padding_down: int = diff_down if (diff_down > 0) else 0

    padding = ConstantPad2d((padding_left, padding_right, padding_up, padding_down), board.tiles_populated["wall_tile"].visual)
    return flatten(padding(clipped_tensor))

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
