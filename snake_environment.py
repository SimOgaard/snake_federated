### TODO:
###     Super fruit that gives more reward but gives you longer body
###     Flatten tensor before sending through nn
###     its possible for fruit to be placed on snake body (it should not be possible)
###
###     random board structure
###     
###     stack board tensor when epoch % 1000 == 0 and save it

# dataclass (c# struct like equivalent)
from dataclasses import dataclass
from dataclasses import field

# pytorch tensors
from torch import empty as torch_empty

# math functions
from numpy.random import rand
from numpy import True_, array, empty
from math import floor
from random import randrange
from random import choice

# dataclasses
@dataclass
class Rewards():
    ''' Dataclass holding all rewards '''
    snake: float = -999
    edge: float = -100
    mine: float = -25
    food: float = 555
    air: float = -1

@dataclass
class Board():
    ''' Dataclass holding structure for game board '''
    max_board_shape: array = array([10, 10])
    min_board_shape: array = array([10, 10])

    salt_and_pepper_chance: float = 0.1
    food_amount: int = 1 # how many foods that should be placed

    # data placeholders (gets assigned a value in code)
    board_shape: array = None # 19 rows with 9 columns
    board = None # torch.FloatTensor
    board_on_hit: array = None

    board_shape_no_border: array = None
    open_board_positions: dict = field(default_factory=dict)

@dataclass
class Snake():
    ''' Dataclass holding snake values '''
    snake_body: list = field(default_factory=list) # You need to input snake body as coordinates
    snake_move_count: int = 2
    # data placeholders (gets assigned a value in code)
    snake_direction: array = None # y direction, x direction

class SnakeEnv():
    ''' Environment that agent should act in to get a reward '''

    # tile functions
    def snake_tile(self, **kwargs) -> float:
        self.done = True
        return self.rewards_data.snake

    def edge_tile(self, **kwargs) -> float:
        self.done = True
        return self.rewards_data.edge

    def mine_tile(self, **kwargs) -> float:
        self.remove_snake_tail()
        if len(self.snake_data.snake_body) == 0:
            self.done = True
        return self.rewards_data.mine
        
    def food_tile(self, **kwargs) -> float:
        def get_nth_key(n=0):
            if n < 0:
                n += len(self.board_data.open_board_positions)
            for i, key in enumerate(self.board_data.open_board_positions.keys()):
                if i == n:
                    return key
            raise IndexError("dictionary index out of range")

        # add snake part
        if ("old_snake_tail_pos" in kwargs):
            self.place_new_snake_tail(kwargs["old_snake_tail_pos"])

        # place it on board
        if (self.food_index >= len(self.rand_food_array)):
            self.food_index += 1
            if (self.food_index - self.board_data.food_amount >= len(self.rand_food_array)):
                self.done = True
            return

        # get random coord viable for food placement
        random_food_index: int = floor(self.rand_food_array[self.food_index] * len(self.board_data.open_board_positions))
        random_coord: array = get_nth_key(random_food_index)

        # remove random_coord from open_board_positions dictionary
        del self.board_data.open_board_positions[tuple(random_coord)]

        self.board_data.board[random_coord[0]][random_coord[1]] = self.rewards_data.food
        self.board_data.board_on_hit[random_coord[0]][random_coord[1]] = self.food_tile

        # # add this_tile_pos too open_board_positions dictionary
        # if ("this_tile_pos" in kwargs):
        #     self.board_data.open_board_positions[tuple(kwargs["this_tile_pos"])] = kwargs["this_tile_pos"]

        self.food_index += 1

        return self.rewards_data.food

    def air_tile(self, **kwargs) -> float:
        return self.rewards_data.air

    def __init__(self, rewards_data: Rewards, board_data: Board) -> None:
        ### init rewards data
        self.rewards_data = rewards_data

        ### init board
        self.board_data = board_data

        width: int = randrange(self.board_data.min_board_shape[0], self.board_data.max_board_shape[0]) if self.board_data.min_board_shape[0] != self.board_data.max_board_shape[0] else self.board_data.min_board_shape[0]
        height: int = randrange(self.board_data.min_board_shape[1], self.board_data.max_board_shape[1]) if self.board_data.min_board_shape[1] != self.board_data.max_board_shape[1] else self.board_data.min_board_shape[1]
        
        self.board_data.board_shape = array([width, height])

        self.board_data.board_shape_no_border = self.board_data.board_shape - array([2, 2])
        self.board_data.board = torch_empty(tuple(self.board_data.board_shape))
        self.board_data.board_on_hit = empty(tuple(self.board_data.board_shape), dtype=object)

        # place values on board that represent the reward and method that should be given if snake head goes on it
        rand_salt_and_pepper: array = rand(self.board_data.board_shape[0] * self.board_data.board_shape[1])

        for row in range(self.board_data.board_shape[0]):
            for col in range(self.board_data.board_shape[1]):
                is_side: bool = row == 0 or row == self.board_data.board_shape[0] - 1
                is_top: bool = col == 0 or col == self.board_data.board_shape[1] - 1

                salt_and_pepper: bool = rand_salt_and_pepper[row * self.board_data.board_shape[1] + col] < self.board_data.salt_and_pepper_chance

                if (is_side or is_top):
                    self.board_data.board[row][col] = self.rewards_data.edge
                    self.board_data.board_on_hit[row][col] = self.edge_tile
                elif salt_and_pepper:
                    self.board_data.board[row][col] = self.rewards_data.mine
                    self.board_data.board_on_hit[row][col] = self.mine_tile
                else:
                    self.board_data.board[row][col] = self.rewards_data.air
                    self.board_data.board_on_hit[row][col] = self.air_tile

                    self.board_data.open_board_positions[(row, col)] = array([row, col])
    
class SnakeAgent(SnakeEnv):
    ''' Our agent '''
    def create_snake(self) -> None:
        # choose random point inside board_shape
        start_coord: array
        tail_coord: array

        while True:
            start_coord = array([randrange(1, self.board_data.board_shape[0] - 2), randrange(1, self.board_data.board_shape[1] - 2)])
            if self.board_data.board[start_coord[0]][start_coord[1]] == self.rewards_data.air:
                for _try in range(10):
                    tail_offset: array = array([0, 0])
                    array_index: int = randrange(1)
                    tail_offset[array_index] = choice([-1, 1])

                    tail_coord = start_coord - tail_offset
                    if self.board_data.board[tail_coord[0]][tail_coord[1]] == self.rewards_data.air:
                        break
                
                return Snake([start_coord, tail_coord])

    def remove_snake_tail(self) -> array:
        tail_point: array = self.snake_data.snake_body.pop()
        self.board_data.board[tail_point[0]][tail_point[1]] = self.rewards_data.air
        self.board_data.open_board_positions[(tail_point[0], tail_point[1])] = tail_point
        self.board_data.board_on_hit[tail_point[0]][tail_point[1]] = self.air_tile
        return tail_point

    def place_new_snake_head(self, snake_part: array) -> None:
        self.snake_data.snake_body.insert(0, snake_part)
        self.place_snake_tile(snake_part)

    def place_new_snake_tail(self, snake_part: array) -> None:
        self.snake_data.snake_body.append(snake_part)
        self.place_snake_tile(snake_part)

    def place_snake_tile(self, snake_part: array) -> None:
        # place snake reward on board
        self.board_data.board[snake_part[0]][snake_part[1]] = self.rewards_data.snake
        # place snake method on board
        self.board_data.board_on_hit[snake_part[0]][snake_part[1]] = self.snake_tile
        # remove point from dictionary so nothing can be placed there
        if (tuple(snake_part) in self.board_data.open_board_positions):
            del self.board_data.open_board_positions[tuple(snake_part)]

    def __init__(self, rewards_data: Rewards, board_data: Board) -> None:
        ### init env
        super().__init__(rewards_data, board_data)
        
        ### init snake
        self.done: bool = False
        self.snake_data = self.create_snake()
        # get direction of snake
        self.snake_data.snake_direction = -(self.snake_data.snake_body[-1] - self.snake_data.snake_body[-2])

        # set snake body
        for snake_part in self.snake_data.snake_body:
            self.place_snake_tile(snake_part)        

        ### food
        self.food_index = 0
        self.rand_food_array = rand(len(self.board_data.open_board_positions))

        for _ in range(self.board_data.food_amount):
            self.food_tile()

    def move_snake(self, snake_direction: array) -> float:
        # set direction if dot product is not negative
        if ((snake_direction != -self.snake_data.snake_direction).all()):
            self.snake_data.snake_direction = snake_direction

        reward_sum: float = 0

        for i in range(self.snake_data.snake_move_count):
            moved_head_point: array = self.snake_data.snake_body[0] + self.snake_data.snake_direction

            # move tail
            tail_point: array = self.remove_snake_tail()
            
            # check for snake head collision
            reward_sum += float(self.board_data.board[moved_head_point[0]][moved_head_point[1]])
            collision_method: function = self.board_data.board_on_hit[moved_head_point[0]][moved_head_point[1]]

            self.place_new_snake_head(moved_head_point)
            collision_method(this_tile_pos = moved_head_point, old_snake_tail_pos = tail_point)
            
            if (self.done):
                break
        return reward_sum