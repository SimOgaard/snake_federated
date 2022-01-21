# Math modules
from numpy import array
from numpy.linalg import norm

# Torch imports
from torch import FloatTensor, flatten, cat, where
from torch.nn import ConstantPad2d

# Repo imports
from snake_env.snake_agents.virtual_snake import Snake
from snake_env.snake_environment import Board

def observation_full(board: Board, *args, **kwargs) -> FloatTensor:
    '''
    Returns the whole board
    '''
    return flatten(board.board.detach().clone())

def observation_near(snake: Snake, kernel: array, *args, **kwargs) -> FloatTensor:
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
    
    # get offset from snake head in all directions
    x_y_offset = (kernel - array([1, 1])) // 2
    snake_head: array = snake.snake_body[0]

    # get x coordinates 
    x_from: int = snake_head[0] - x_y_offset[0]
    x_from_clamped: int = clamp_coord(x_from, 0, snake.board.max_board_shape[0] + 1)
    x_to: int = snake_head[0] + x_y_offset[0] + 1
    x_to_clamped: int = clamp_coord(x_to, 0, snake.board.max_board_shape[0] + 2)

    # get y coordinates 
    y_from: int = snake_head[1] - x_y_offset[1]
    y_from_clamped: int = clamp_coord(y_from, 0, snake.board.max_board_shape[1] + 1)
    y_to: int = snake_head[1] + x_y_offset[1] + 1
    y_to_clamped: int = clamp_coord(y_to, 0, snake.board.max_board_shape[1] + 2)

    # select board 
    clipped_tensor = snake.board.board[x_from_clamped:x_to_clamped,y_from_clamped:y_to_clamped]

    # get padding values
    diff_left: int = y_from_clamped - y_from
    padding_left: int = diff_left if (diff_left > 0) else 0
    diff_right: int = y_to - y_to_clamped
    padding_right: int = diff_right if (diff_right > 0) else 0
    diff_up: int = x_from_clamped - x_from
    padding_up: int = diff_up if (diff_up > 0) else 0
    diff_down: int = x_to - x_to_clamped
    padding_down: int = diff_down if (diff_down > 0) else 0

    # pad tensor
    padding = ConstantPad2d((padding_left, padding_right, padding_up, padding_down), snake.board.tiles_populated["wall_tile"].visual)
    return flatten(padding(clipped_tensor))

def observation_to_bool(tensor: FloatTensor) -> FloatTensor:
    '''
    Returns bool tensor of input tensor
    '''
    return where(tensor > 0, 1., 0.)

def observation_food(snake: Snake, *args, **kwargs) -> FloatTensor:
    '''
    Returns float tensor where nearest food is [down, up, right, left] tiles from snake_head
    '''
    if (len(snake.board.all_food_on_board) == 0):
        return FloatTensor([0, 0, 0, 0])

    snake_head: array = snake.snake_body[0]

    dist: float = float('inf')
    closest_food: array
    for food in snake.board.all_food_on_board:
        new_dist: float = norm(food - snake_head)
        if new_dist < dist:
            closest_food = food
            if (new_dist == 1.):
                break

    closest_food -= snake_head

    down: float = closest_food[0] if closest_food[0] > 0 else 0
    up: float = -closest_food[0] if closest_food[0] < 0 else 0

    right: float = closest_food[1] if closest_food[1] > 0 else 0
    left: float = -closest_food[1] if closest_food[1] < 0 else 0

    return FloatTensor([down, up, right, left])

def observation_cat(*args):
    '''
    Cats given tensors into a 1d tensor
    '''
    return cat(args)