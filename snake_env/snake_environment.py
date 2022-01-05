# Torch imports
from numpy.lib.shape_base import tile
from torch import empty as torch_empty
from torch import tensor
from torch import float as torch_float

# Math modules
from numpy.random import rand
from numpy import empty
from random import randrange
from random import randint

# Repo imports
from snake_env.tiles.tiles import *
from snake_env.tiles.tiles_spawn import *

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

        self.set_snakes(snakes)

        self.tiles_populated = tiles_populated
        # needs to run __restart__ for board to start working, is done externally
        # self.__restart__()

    def set_snakes(self, snakes: list) -> None:
        self.snakes = snakes
        for snake in self.snakes:
            snake.board = self

    def __restart__(self) -> None:
        '''
        Restarts board for new run (need to be executed before each new run including the first)
        '''
        super(TilesSpawn, self).__init__()

        self.open_board_positions = {} # why cant python fuck fase fuxkcklahsjd fhjkl init capacity of this dict?!?!

        assert self.min_board_shape[0] <= self.max_board_shape[0]
        assert self.min_board_shape[1] <= self.max_board_shape[1]

        # true_board_width: int = self.max_board_shape[0] + 2
        # true_board_height: int = self.max_board_shape[1] + 2

        width: int = randint(self.min_board_shape[0], self.max_board_shape[0])
        height: int = randint(self.min_board_shape[1], self.max_board_shape[1])
        start_row: int = randint(0, self.max_board_shape[0] - width) + 1
        start_col: int = randint(0, self.max_board_shape[1] - height) + 1

        self.bounding_box = array([start_row, start_col, start_row + width, start_col + height])
        
        # ## THIS IS SLOW AF
        # import time
        # start = time.time()
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
        
        # ugly ass Python way is fast because its compiled to an acutal language c:

        # fill both tensor and np array with air
        self.board_tiles.fill(self.tiles_populated["air_tile"])
        self.board.fill_(self.tiles_populated["air_tile"].visual)

        # fill sides by boundingbox indices with wall
        # tensor:
        indices_row: list = list(range(0, self.bounding_box[0])) + list(range(self.bounding_box[2], self.board.shape[0]))
        self.board.index_fill_(0, tensor(indices_row), self.tiles_populated["wall_tile"].visual)
        indices_col = list(range(0, self.bounding_box[1])) + list(range(self.bounding_box[3], self.board.shape[1]))
        self.board.index_fill_(1, tensor(indices_col), self.tiles_populated["wall_tile"].visual)
        # numpy:
        self.board_tiles[indices_row,:] = self.tiles_populated["wall_tile"]
        self.board_tiles[:, indices_col] = self.tiles_populated["wall_tile"]
        
        # end = time.time()
        # print("c implementation:" + str(end - start))
        # start = time.time()
        # fill dictionary:
        for row in range(self.bounding_box[0], self.bounding_box[2]):
            for col in range(self.bounding_box[1], self.bounding_box[3]):
                self.open_board_positions[(row, col)] = array([row, col])
        # end = time.time()
        # print("python itteration that i do not know how to fix:" + str(end - start))
        self.run += 1

        self.temporary_snakes: list = []

        # place snake
        for snake in self.snakes:
            # init snake
            snake.__restart__()

        for tile in self.tiles_populated.values():
            self.spawn_tile(tile)

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
