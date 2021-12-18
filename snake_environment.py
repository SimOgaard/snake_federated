### TODO:
###     Super fruit that gives more reward but gives you longer body
###     Flatten tensor before sending through nn
###     its possible for fruit to be placed on snake body (it should not be possible)
###
###     random board structure
###     
###     stack board tensor when epoch % 1000 == 0 and save it

# dataclass (c# struct like equivalent)
from dataclasses import dataclass, field

# pytorch tensors
from torch import empty as torch_empty
from torch import FloatTensor, clone

# math functions
from numpy.random import rand
from numpy import array, empty
from math import floor
from random import randrange, shuffle

class Tiles():
    '''
    Tile functions
    '''

    # dataclass for rewards
    @dataclass
    class TileRewards():
        '''
        Dataclass holding all rewards
        '''
        snake: float = -1111
        edge: float = -1000
        mine: float = -25
        food: float = 1000
        air: float = -10

    # dataclass for visual
    @dataclass
    class TileVisual():
        '''
        Dataclass holding all rewards
        '''
        snake_head: float = 5.
        snake: float = 4.
        edge: float = 3.
        mine: float = 2.
        food: float = 1.
        air: float = 0.

    # tiles init
    def __init__(self) -> None:
        super(Tiles, self).__init__()

    # tiles restart
    def __restart__(self) -> None:
        # super(Tiles, self).__restart__()
        pass

    # returns random tilecord that is not occupied
    def random_open_tile_coord(self) -> array:
        def get_nth_key(n=0):
            if n < 0:
                n += len(self.board_data.open_board_positions)
            for i, key in enumerate(self.board_data.open_board_positions.keys()):
                if (i == n):
                    return key
            raise IndexError("dictionary index out of range")
        
        random_index: int = floor(randrange(len(self.board_data.open_board_positions)))
        return get_nth_key(random_index)

    # places given tile at given position on board
    def place_tile(self, coord:array, tile_function:object, tile_visual:int, occupy:bool) -> None:
        self.board_data.board[coord[0]][coord[1]] = tile_visual
        self.board_data.board_on_hit[coord[0]][coord[1]] = tile_function
        
        if (occupy):
            if (tuple(coord) in self.board_data.open_board_positions):
                del self.board_data.open_board_positions[tuple(coord)]
        else:
            self.board_data.open_board_positions[tuple(coord)] = coord

    def air_tile(self, **kwargs) -> float:
        return (Tiles.TileRewards.air, Tiles.TileVisual.air)

    def edge_tile(self, **kwargs) -> tuple:
        self.done = True
        return (Tiles.TileRewards.edge, Tiles.TileVisual.edge)

    def snake_tile(self, **kwargs) -> tuple:
        self.done = True
        return (Tiles.TileRewards.snake, Tiles.TileVisual.snake)

    def mine_tile(self, **kwargs) -> tuple:
        self.remove_snake_tail()
        if (len(self.snake_body) == 0):
            self.done = True
        return (Tiles.TileRewards.mine, Tiles.TileVisual.mine)
        
    def food_tile(self, **kwargs) -> tuple:
        # add snake part
        if ("old_snake_tail_pos" in kwargs):
            self.place_new_snake_tail(kwargs["old_snake_tail_pos"])

        # place new food on board
        if (len(self.board_data.open_board_positions) != 0):
            # get random coord viable for food placement
            random_coord: array = self.random_open_tile_coord()
            self.place_tile(random_coord, self.food_tile, Tiles.TileVisual.food, True)
        return (Tiles.TileRewards.food, Tiles.TileVisual.food)

class Board(Tiles):
    '''
    Environment that agent should act in to get a reward
    '''

    @dataclass
    class BoardData():
        '''
        Dataclass holding structure for game board
        '''
        min_board_shape: array
        max_board_shape: array

        salt_and_pepper_chance: float
        food_amount: int

        replay_interval: int

        # data placeholders (gets assigned a value in code)
        bounding_box: array = None
        board: FloatTensor = None
        board_on_hit: array = None
        board_replay: list = None

        open_board_positions: dict = field(default_factory=dict)

    # board init
    def __init__(self, board_data) -> None:
        # init tiles
        super(Board, self).__init__()

        # init board data
        self.board_data = board_data

        self.board_data.board = torch_empty(tuple(self.board_data.max_board_shape + array([2, 2])))
        self.board_data.board_on_hit = empty(tuple(self.board_data.max_board_shape + array([2, 2])), dtype=object)

        # 3d cube of runs
        self.board_data.board_replay = []

        # needs to run restart but you do it in agent
        # self.__restart__()
    
    # board restart
    def __restart__(self) -> None:

        def better_rand(x:int, y:int):
            if (x != y):
                return randrange(x, y)
            return x

        # restart tiles
        super(Board, self).__restart__()

        self.board_data.open_board_positions = {}

        assert self.board_data.min_board_shape[0] <= self.board_data.max_board_shape[0]
        assert self.board_data.min_board_shape[1] <= self.board_data.max_board_shape[1]

        true_board_width: int = self.board_data.max_board_shape[0] + 2
        true_board_height: int = self.board_data.max_board_shape[1] + 2

        width: int = better_rand(self.board_data.min_board_shape[0], self.board_data.max_board_shape[0])
        height: int = better_rand(self.board_data.min_board_shape[1], self.board_data.max_board_shape[1])
        start_row: int = better_rand(0, self.board_data.max_board_shape[0] - width) + 1
        start_col: int = better_rand(0, self.board_data.max_board_shape[1] - height) + 1

        self.board_data.bounding_box = array([start_row, start_col, start_row + width, start_col + height])
        
        rand_salt_and_pepper: array = rand(true_board_width * true_board_height)

        for row in range(true_board_width):
            for col in range(true_board_height):
                is_side: bool = row < self.board_data.bounding_box[0] or row > self.board_data.bounding_box[2]
                is_top: bool = col < self.board_data.bounding_box[1] or col > self.board_data.bounding_box[3]

                salt_and_pepper: bool = rand_salt_and_pepper[row * true_board_height + col] < self.board_data.salt_and_pepper_chance

                if (is_side or is_top):
                    self.board_data.board[row][col] = Tiles.TileVisual.edge
                    self.board_data.board_on_hit[row][col] = self.edge_tile
                elif salt_and_pepper:
                    self.board_data.board[row][col] = Tiles.TileVisual.mine
                    self.board_data.board_on_hit[row][col] = self.mine_tile
                else:
                    self.place_tile(array([row, col]), self.air_tile, Tiles.TileVisual.air, False)

class Snake(Board):
    '''
    Our snake agent
    '''

    @dataclass
    class SnakeData():
        '''
        Dataclass holding snake values
        '''

        snake_body: list = field(default_factory=list) # You need to input snake body as coordinates
        snake_move_count: int = 1
        
        # data placeholders (gets assigned a value in code)
        snake_direction: array = None # y direction, x direction

    # snake init
    def __init__(self, board_data: Board.BoardData) -> None:
        ### init board
        super(Snake, self).__init__(board_data)
        self.run = 0

        # needs to run restart but you do it in agent
        # self.__restart__()

    # snake restart
    def __restart__(self) -> None:
        # restart board
        super(Snake, self).__restart__()

        self.done: bool = False
        self.snake_data = self.create_snake()
        # get direction of snake
        self.snake_data.snake_direction = -(self.snake_data.snake_body[-1] - self.snake_data.snake_body[-2])
        # set snake body
        self.place_tile(self.snake_data.snake_body[0], self.snake_tile, Tiles.TileVisual.snake_head, True)
        for snake_part in self.snake_data.snake_body[1:]:
            self.place_tile(snake_part, self.snake_tile, Tiles.TileVisual.snake, True)

        for _ in range(self.board_data.food_amount):
            self.food_tile()

    def create_snake(self) -> None:
        # choose random point inside board_shape
        start_coord: array
        tail_coord: array
        offset_sequence: list = [array([1, 0]), array([-1, 0]), array([0, 1]), array([0, -1])]

        while True:
            start_coord = array([randrange(self.board_data.bounding_box[0], self.board_data.bounding_box[2]), randrange(self.board_data.bounding_box[1], self.board_data.bounding_box[3])])
            if self.board_data.board[start_coord[0]][start_coord[1]] == Tiles.TileVisual.air:
                shuffle(offset_sequence)
                for tail_offset in offset_sequence:
                    tail_coord = start_coord - tail_offset
                    if self.board_data.board[tail_coord[0]][tail_coord[1]] == Tiles.TileVisual.air:
                        break
                
                return Snake.SnakeData([start_coord, tail_coord])

    def remove_snake_tail(self) -> array:
        tail_point: array = self.snake_data.snake_body.pop()
        self.place_tile(tail_point, self.air_tile, Tiles.TileVisual.air, False)
        return tail_point

    def place_new_snake_head(self, snake_coord: array) -> None:
        self.place_tile(self.snake_data.snake_body[0], self.snake_tile, Tiles.TileVisual.snake, True)
        self.snake_data.snake_body.insert(0, snake_coord)
        self.place_tile(snake_coord, self.snake_tile, Tiles.TileVisual.snake_head, True)
        
    def place_new_snake_tail(self, snake_coord: array) -> None:
        self.snake_data.snake_body.append(snake_coord)
        self.place_tile(snake_coord, self.snake_tile, Tiles.TileVisual.snake, True)

    def move_snake(self, snake_direction: array) -> float:
        # set direction if dot product is not negative
        if ((snake_direction != -self.snake_data.snake_direction).all()):
            self.snake_data.snake_direction = snake_direction

        reward_sum: float = 0

        if (self.run % self.board_data.replay_interval == 0):
            self.board_data.board_replay.append(clone(self.board_data.board))

        for i in range(self.snake_data.snake_move_count):
            moved_head_point: array = self.snake_data.snake_body[0] + self.snake_data.snake_direction

            # move tail
            tail_point: array = self.remove_snake_tail()
            
            # check for snake head collision
            collision_method: function = self.board_data.board_on_hit[moved_head_point[0]][moved_head_point[1]]

            self.place_new_snake_head(moved_head_point)
            reward_sum += collision_method(this_tile_pos = moved_head_point, old_snake_tail_pos = tail_point)[0]
            
            if (self.done):
                if (self.run % self.board_data.replay_interval == 0):
                    self.board_data.board_replay.append(clone(self.board_data.board))

                self.run += 1
                break

        return reward_sum

class Agent(Snake):
    pass