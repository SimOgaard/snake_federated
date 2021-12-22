# Torch imports
from torch import empty as torch_empty

# Math modules
from numpy.random import rand
from numpy import empty
from random import randrange

# Repo imports
from snake_env.tiles.tiles import *

class Board():
    '''
    Snake game board
    '''
    def __init__(self, min_board_shape: array, max_board_shape: array, salt_and_pepper_chance: float, food_amount: array, replay_interval: int, snakes: list) -> None:
        '''
        Initilizes a Board object 
        '''
        super(Board, self).__init__()

        # init board data
        self.min_board_shape: array = min_board_shape
        self.max_board_shape: array = max_board_shape

        self.salt_and_pepper_chance: float = salt_and_pepper_chance
        self.food_amount: array = food_amount

        self.replay_interval: int = replay_interval

        self.bounding_box: array = None
        self.board = torch_empty(tuple(self.max_board_shape + array([2, 2])))
        self.board_tiles = empty(tuple(self.max_board_shape + array([2, 2])), dtype=object)

        # 3d cube of runs
        self.run: int = -1
        self.board_replay: list = []

        self.snakes = snakes
        for snake in self.snakes:
            snake.board = self

        # needs to run __restart__ for board to start working, is done externally
        # self.__restart__()
    
    def __restart__(self) -> None:
        '''
        Restarts board for new run (need to be executed before each new run including the first)
        '''
        def better_rand(x: int, y: int):
            '''
            Workaround this stupid fucking code snippet: assert x != y in random.randrange
            '''
            if (x != y):
                return randrange(x, y)
            return x

        self.open_board_positions = {}

        assert self.min_board_shape[0] <= self.max_board_shape[0]
        assert self.min_board_shape[1] <= self.max_board_shape[1]

        true_board_width: int = self.max_board_shape[0] + 2
        true_board_height: int = self.max_board_shape[1] + 2

        width: int = better_rand(self.min_board_shape[0], self.max_board_shape[0])
        height: int = better_rand(self.min_board_shape[1], self.max_board_shape[1])
        start_row: int = better_rand(0, self.max_board_shape[0] - width) + 1
        start_col: int = better_rand(0, self.max_board_shape[1] - height) + 1

        self.bounding_box = array([start_row, start_col, start_row + width - 1, start_col + height - 1])
        
        for row in range(true_board_width):
            for col in range(true_board_height):
                is_side: bool = row < self.bounding_box[0] or row > self.bounding_box[2]
                is_top: bool = col < self.bounding_box[1] or col > self.bounding_box[3]

                if (is_side or is_top):
                    self.board_tiles[row][col] = WallTile()
                else:
                    coord: array =([row, col])
                    self.open_board_positions[tuple(coord)] = coord
                    self.board_tiles[row][col] = AirTile()

                self.board[row][col] = self.board_tiles[row][col].visual
        
        self.run += 1

        self.temporary_snakes: list = []

        # place snake
        for snake in self.snakes:
            # init snake
            snake.__restart__()
            # Places snake body on board
            self.place_tile(SnakeHeadTile(), snake.snake_body[0])
            for snake_part in snake.snake_body[1:]:
                self.place_tile(SnakeTile(), snake_part)

        # for every open position on board
        rand_salt_and_pepper: array = rand(len(self.open_board_positions))
        for index, coord in enumerate(self.open_board_positions.values()):
            if rand_salt_and_pepper[index] < self.salt_and_pepper_chance:
                self.place_tile(MineTile(), coord)

        for _ in range(better_rand(self.food_amount[0], self.food_amount[1])):
            if (len(self.open_board_positions) == 0):
                break
            # get random coord viable for food placement
            random_coord: array = self.random_open_tile_coord()
            self.place_tile(FoodTile(), random_coord)

        return self

    def is_alive(self) -> bool:
        '''
        Are all snakes alive?
        '''

        if (self.replay_interval != 0 and self.run % self.replay_interval == 0):
            self.board_replay.append(self.board.detach().clone())

        # make all temporary snakes act and move
        for snake in self.temporary_snakes:
            if (not snake.done):
                action: int = snake.act()
                snake.move(action)

        snakes_alive: int = 0
        for snake in self.snakes:
            if (not snake.done):
                snakes_alive += 1

        return snakes_alive != 0

    def random_open_tile_coord(self) -> array:
        '''
        Returns random tilecord that is not occupied by tiles marked with occupy
        '''
        def get_nth_key(n=0):
            if n < 0:
                n += len()
            for i, key in enumerate(self.open_board_positions.keys()):
                if (i == n):
                    return key
            raise IndexError("dictionary index out of range")
        
        random_index: int = randrange(len(self.open_board_positions))
        return get_nth_key(random_index)

    def place_tile(self, tile: Tiles, coord: array) -> None:
        '''
        Places tile at given position on board
        '''
        self.board_tiles[coord[0]][coord[1]] = tile
        self.board[coord[0]][coord[1]] = tile.visual
        
        if (tile.occupy):
            if (tuple(coord) in self.open_board_positions):
                del self.open_board_positions[tuple(coord)]
        else:
            self.open_board_positions[tuple(coord)] = coord