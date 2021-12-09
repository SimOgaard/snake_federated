### TODO:
###     Super fruit that gives more reward but gives you longer body
###     Flatten tensor before sending through nn
###     its possible for fruit to be placed on snake body (it should not be possible)
###
###     random board structure

# dataclass (c# struct like equivalent)
from dataclasses import dataclass
from dataclasses import field

# pytorch tensors
from torch import empty as torch_empty

# math functions
from numpy.random import rand
from numpy import array, empty
from math import floor
from random import randrange
from random import choice

# dataclasses
@dataclass
class Rewards():
    ''' Dataclass holding all rewards '''
    snake: float = -999
    edge: float = -100
    food: float = 555
    air: float = -1

@dataclass
class Board():
    ''' Dataclass holding structure for game board '''
    max_board_shape: array = array([20, 11])
    min_board_shape: array = array([4, 4])

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
    snake_body: list[array] # You need to input snake body as coordinates

    # data placeholders (gets assigned a value in code)
    snake_direction: array = None # y direction, x direction

class SnakeEnv():
    ''' Environment that agent should act in to get a reward '''

    # tile functions
    def snake_tile(self, **kwargs) -> float:
        self.done = True
        print("snake_tile")
        return self.rewards_data.snake

    def edge_tile(self, **kwargs) -> float:
        self.done = True
        print("edge_tile")
        return self.rewards_data.edge
        
    def food_tile(self, **kwargs) -> float:
        if (self.food_index >= len(self.rand_food_array)):
            self.done = True
            return

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

        # get random coord viable for food placement
        random_food_index: int = floor(self.rand_food_array[self.food_index] * len(self.board_data.open_board_positions))
        random_coord: array = get_nth_key(random_food_index)

        # remove random_coord from open_board_positions dictionary
        del self.board_data.open_board_positions[tuple(random_coord)]

        # place it on board
        self.board_data.board[random_coord[0]][random_coord[1]] = self.rewards_data.food
        self.board_data.board_on_hit[random_coord[0]][random_coord[1]] = self.food_tile

        # # add this_tile_pos too open_board_positions dictionary
        # if ("this_tile_pos" in kwargs):
        #     self.board_data.open_board_positions[tuple(kwargs["this_tile_pos"])] = kwargs["this_tile_pos"]

        self.food_index += 1

        print("food_tile")
        return self.rewards_data.food

    def air_tile(self, **kwargs) -> float:
        print("air_tile")
        return self.rewards_data.air

    def __init__(self, rewards_data: Rewards, board_data: Board) -> None:
        ### init rewards data
        self.rewards_data = rewards_data

        ### init board
        self.board_data = board_data

        width: int = randrange(self.board_data.min_board_shape[0], self.board_data.max_board_shape[0])
        height: int = randrange(self.board_data.min_board_shape[1], self.board_data.max_board_shape[1])
        self.board_data.board_shape = array([width, height])

        self.board_data.board_shape_no_border = self.board_data.board_shape - array([2, 2])
        self.board_data.board = torch_empty(tuple(self.board_data.board_shape))
        self.board_data.board_on_hit = empty(tuple(self.board_data.board_shape), dtype=object)

        # place values on board that represent the reward and method that should be given if snake head goes on it
        rand_salt_and_pepper: array = rand(self.board_data.board_shape[0] * self.board_data.board_shape[1])
        print(len(rand_salt_and_pepper))

        for row in range(self.board_data.board_shape[0]):
            for col in range(self.board_data.board_shape[1]):
                is_side: bool = row == 0 or row == self.board_data.board_shape[0] - 1
                is_top: bool = col == 0 or col == self.board_data.board_shape[1] - 1

                salt_and_pepper: bool = rand_salt_and_pepper[row * self.board_data.board_shape[1] + col] < self.board_data.salt_and_pepper_chance

                if (is_side or is_top or salt_and_pepper):
                    self.board_data.board[row][col] = self.rewards_data.edge
                    self.board_data.board_on_hit[row][col] = self.edge_tile
                else:
                    self.board_data.board[row][col] = self.rewards_data.air
                    self.board_data.board_on_hit[row][col] = self.air_tile

                    self.board_data.open_board_positions[(row, col)] = array([row, col])
    
class SnakeAgent(SnakeEnv):
    ''' Our agent '''
    def create_snake(self) -> None:
        # choose random point inside board_shape
        start_coord: array
        while True:
            start_coord = array([randrange(1, self.board_data.board_shape[0] - 2), randrange(1, self.board_data.board_shape[1] - 2)])
            if self.board_data.board[start_coord[0]][start_coord[1]] == self.rewards_data.air:
                break
        
        tail_coord: array
        while True:
            tail_offset: array = array([0, 0])
            array_index: int = randrange(1)
            tail_offset[array_index] = choice([-1, 1])

            tail_coord = start_coord - tail_offset
            if self.board_data.board[tail_coord[0]][tail_coord[1]] == self.rewards_data.air:
                break
        
        return Snake([start_coord, tail_coord])

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
        self.food_tile()

        #print(chr(27) + "[2J")
        print(self.board_data.board)

    def move_snake(self, snake_direction: array) -> float:
        # clear terminal
        print(chr(27) + "[2J")

        # set direction if dot product is not negative
        if ((snake_direction != -self.snake_data.snake_direction).all()):
            self.snake_data.snake_direction = snake_direction

        # move tail
        tail_point: array = self.snake_data.snake_body.pop()
        self.board_data.board[tail_point[0]][tail_point[1]] = self.rewards_data.air
        self.board_data.open_board_positions[(tail_point[0], tail_point[1])] = tail_point
        self.board_data.board_on_hit[tail_point[0]][tail_point[1]] = self.air_tile
        
        # check for snake head collision
        moved_head_point: array = self.snake_data.snake_body[0] + self.snake_data.snake_direction
        reward: float = float(self.board_data.board[moved_head_point[0]][moved_head_point[1]])
        collision_method: function = self.board_data.board_on_hit[moved_head_point[0]][moved_head_point[1]]

        self.place_new_snake_head(moved_head_point)
        collision_method(this_tile_pos = moved_head_point, old_snake_tail_pos = tail_point)

        # print board
        print(self.board_data.board)
        #print(self.board_data.board_on_hit)
        
        return reward
    
# input testing
from msvcrt import getch
def KeyCheck() -> array:
    global Break_KeyCheck
    Break_KeyCheck = False
    
    base = getch()
    if base == b'\x00':
        sub = getch()
        
        if sub == b'H':
            return array([-1, 0])

        elif sub == b'M':
            return array([0, 1])

        elif sub == b'P':
            return array([1, 0])

        elif sub == b'K':
            return array([0, -1])


# you should be able to roll back in time
if __name__ == "__main__":
    reward_data: Rewards = Rewards()
    board_data: Board = Board()
    # snake_data: Snake = Snake([array([2, 1]), array([3, 1]), array([4, 1])])
    #snake_data: Snake = Snake.create_snake(board_data)
    #print(snake_data.snake_body)
    snake: SnakeAgent = SnakeAgent(reward_data, board_data)

    all_reward: int = 0
    while True:
        player_input: array = KeyCheck()

        this_reward: float = snake.move_snake(player_input)
        all_reward += this_reward
        print(all_reward)

        if (snake.done):
            break


    # input is down left right