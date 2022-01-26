# Torch imports
from torch import empty as torch_empty
from torch import tensor
from torch import float as torch_float

# Math modules
from numpy import empty
from random import randrange
from random import randint

# Repo imports
from snake_env.tiles.tiles import *
from snake_env.tiles.tiles_spawn import *

from itertools import chain

class Board(TilesSpawn):
    '''
    Snake game board
    '''
    def __init__(self, min_board_shape: array, max_board_shape: array, replay_interval: int, snakes: list, tiles_populated: dict = {"air_tile": AirTile(), "wall_tile": WallTile()}) -> None:
        '''
        Initilizes a Board object 
        '''
        super(Board, self).__init__()

        # init board data
        self.min_board_shape: array = min_board_shape
        self.max_board_shape: array = max_board_shape

        self.replay_interval: int = replay_interval

        self.bounding_box: array = None
        self.board = torch_empty(tuple(self.max_board_shape + array([2, 2])), dtype=torch_float)
        self.board_tiles = empty(tuple(self.max_board_shape + array([2, 2])), dtype=object)

        # 3d cube of runs
        self.run: int = -1
        self.board_replay: list = []

        self.set_snakes(*snakes)

        self.tiles_populated = tiles_populated

        self.all_board_positions = {}

        for row in range(0, self.board.shape[0]):
            for col in range(0, self.board.shape[1]):
                self.all_board_positions[(row, col)] = array([row, col])

        # needs to run __restart__ for board to start working, is done externally
        # self.__restart__()

    def set_snakes(self, *arg) -> None:
        '''
        Sets given snakes to be on self
        '''
        self.snakes = list(arg)
        for snake in self.snakes:
            snake.board = self

    def __restart__(self) -> None:
        '''
        Restarts board for new run (need to be executed before each new run including the first)
        '''
        super(TilesSpawn, self).__init__()

        self.all_food_on_board: dict = {}

        assert self.min_board_shape[0] <= self.max_board_shape[0]
        assert self.min_board_shape[1] <= self.max_board_shape[1]

        width: int = randint(self.min_board_shape[0], self.max_board_shape[0])
        height: int = randint(self.min_board_shape[1], self.max_board_shape[1])
        start_row: int = randint(0, self.max_board_shape[0] - width) + 1
        start_col: int = randint(0, self.max_board_shape[1] - height) + 1

        self.bounding_box = array([start_row, start_col, start_row + width, start_col + height])
        
        ### this is programmed c-like (easy to read) but its python; so its slow af
        # for row in range(true_board_width):
        #     for col in range(true_board_height):
        #         is_side: bool = row < self.bounding_box[0] or row > self.bounding_box[2]
        #         is_top: bool = col < self.bounding_box[1] or col > self.bounding_box[3]

        #         if (is_side or is_top):
        #             self.board_tiles[row][col] = self.tiles_populated["wall_tile"]
        #         else:
        #             self.open_board_positions[(row, col)] = array([row, col])
        #             self.board_tiles[row][col] = self.tiles_populated["air_tile"]

        #         self.board[row][col] = self.board_tiles[row][col].visual
        
        ### this is pythonic that is hard to read but is compiled to an acutal language c; so it is faster. eventhough we do a lot of unneeded calculations 🙃
        # fill both tensor and np array with air
        self.board_tiles.fill(self.tiles_populated["air_tile"])
        self.board.fill_(self.tiles_populated["air_tile"].visual)

        # find all indices that should be filled by boundingbox 
        indices_row: list = list(range(0, self.bounding_box[0])) + list(range(self.bounding_box[2], self.board.shape[0]))
        indices_col = list(range(0, self.bounding_box[1])) + list(range(self.bounding_box[3], self.board.shape[1]))

        # tensor:
        self.board.index_fill_(0, tensor(indices_row), self.tiles_populated["wall_tile"].visual)
        self.board.index_fill_(1, tensor(indices_col), self.tiles_populated["wall_tile"].visual)
        # numpy:
        self.board_tiles[indices_row,:] = self.tiles_populated["wall_tile"]
        self.board_tiles[:, indices_col] = self.tiles_populated["wall_tile"]
        
        # fill dictionary (inversed because its faster (fewer closed positions than opened)):
        self.open_board_positions = dict(self.all_board_positions)

        for row in chain(range(0, self.bounding_box[0]), range((self.bounding_box[2]), self.board.shape[0])):
            for col in range(0, self.board.shape[0]):
                self.open_board_positions.pop((row, col))

        for col in chain(range(0, self.bounding_box[1]), range((self.bounding_box[3]), self.board.shape[1])):
            for row in range(self.bounding_box[0], self.bounding_box[2]):
                self.open_board_positions.pop((row, col))

        self.run += 1

        self.temporary_snakes: list = []

        # place snake
        for snake in self.snakes:
            # init snake
            snake.__restart__()

        for tile in self.tiles_populated.values():
            self.spawn_tile(tile)

        self.board_replay: list = []

    def is_alive(self) -> bool:
        '''
        Are all snakes alive?
        '''
        if (self.replay_interval != 0 and self.run % self.replay_interval == 0):
            # save replay
            self.board_replay.append(self.board.detach().clone())

        # make all temporary snakes act and move
        for snake in self.temporary_snakes:
            if (not snake.done):
                action, is_random = snake.act()
                snake.move(action, is_random)

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
            '''
            Indexs into python dictionary is O(n)
            cant index into Python dict wtf?!?!?!?!?!
            '''
            if n < 0:
                n += len()
            for i, value in enumerate(self.open_board_positions.values()):
                if (i == n):
                    return value
            raise IndexError("dictionary index out of range")
        
        random_index: int = randrange(len(self.open_board_positions))
        return get_nth_key(random_index)

    def place_tile(self, tile: Tiles, coord: array) -> None:
        '''
        Places tile at given position on board
        '''
        coord_tuple: tuple = tuple(coord)

        if (isinstance(tile, FoodTile)):
            self.all_food_on_board[coord_tuple] = coord
        elif (isinstance(self.board_tiles[coord[0]][coord[1]], FoodTile)):
            del self.all_food_on_board[coord_tuple]
        
        self.board_tiles[coord[0]][coord[1]] = tile
        self.board[coord[0]][coord[1]] = tile.visual
        
        if (tile.occupy):
            if (coord_tuple in self.open_board_positions):
                del self.open_board_positions[coord_tuple]
        else:
            self.open_board_positions[coord_tuple] = coord
