# Math Modules
from numpy.random import rand
from random import sample
from numpy import array

# Repo imports
from snake_env.tiles.virtual_tiles import Tiles
from generic import *

class TilesSpawn():
    '''
    Spawn functions for tiles
    '''
    def __init__(self) -> None:
        '''
        Initilizes a tiles spawn object
        '''
        super(TilesSpawn, self).__init__()
    
    def __restart__(self) -> None:
        '''
        Initilizes a tiles spawn object
        '''
        pass

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
    
    def get_random_coords(self, amount: int) -> list:
        '''
        Returns random subset of coords that are not occupied
        '''
        return sample(self.open_board_positions.keys(), amount)

    def get_salt_and_pepper(self) -> list:
        '''
        Returns random salt and pepper values for every open board position
        '''
        return rand(len(self.open_board_positions))

    def spawn_tile(self, tile_type: Tiles):
        '''
        Runs every spawn function for specified tile
        '''
        self.spawn_number(tile_type)
        self.spawn_salt_pepper(tile_type)

    def spawn_number(self, tile_type: Tiles):
        '''
        Places a specified number of tiles on board at random places
        '''
        instantiated = tile_type() # instanciate new tile
        spawn_amount = better_rand(instantiated.spawn_amount[0], instantiated.spawn_amount[1])
        for coord in self.get_random_coords(spawn_amount):
            self.place_tile(tile_type(), coord)

    def spawn_salt_pepper(self, tile_type: Tiles):
        '''
        Places specified tile on board at random places
        '''
        instantiated = tile_type() # instanciate new tile
        if (instantiated.salt_pepper_chance > 0):
            for salt_pepper_val, coord in zip(self.get_salt_and_pepper(instantiated.spawn_amount), self.open_board_positions):
                if instantiated.salt_pepper_chance < salt_pepper_val:
                    self.place_tile(tile_type(), coord)

    def spawn_tele_tiles(self, tile_type: Tiles):
        coord_1, coord_2 = self.get_random_coords(2)
        self.place_tile(tile_type(coord_2), coord_1)
        self.place_tile(tile_type(coord_1), coord_2)

