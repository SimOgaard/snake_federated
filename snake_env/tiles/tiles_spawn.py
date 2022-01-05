# Math Modules
from numpy.random import rand
from random import sample
from random import randint

from torch._C import parse_ir

# Repo imports
from snake_env.tiles.virtual_tiles import Tiles
# from generic import *

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
    
    def get_random_coords(self, amount: int) -> list:
        '''
        Returns random subset of coords that are not occupied
        '''
        if amount > len(self.open_board_positions):
            amount = len(self.open_board_positions)
        return sample(self.open_board_positions.keys(), amount)

    def get_salt_and_pepper(self) -> list:
        '''
        Returns random salt and pepper values for every open board position
        '''
        return rand(len(self.open_board_positions))

    def spawn_tile(self, tile: Tiles):
        '''
        Runs every spawn function for specified tile
        '''
        self.spawn_number(tile)
        self.spawn_salt_pepper(tile)
        self.spawn_procent(tile)

    def spawn_number(self, tile: Tiles):
        '''
        Places a specified number of tiles on board at random places
        '''
        spawn_amount: int = randint(tile.spawn_amount[0], tile.spawn_amount[1])
        if (spawn_amount > 0):
            for coord in self.get_random_coords(spawn_amount):
                self.place_tile(tile, coord)

    def spawn_salt_pepper(self, tile: Tiles):
        '''
        Places specified tile on board at random places
        '''
        if (tile.salt_pepper_chance > 0):
            for salt_pepper_val, coord in zip(self.get_salt_and_pepper(), list(self.open_board_positions)):
                if salt_pepper_val < tile.salt_pepper_chance:
                    self.place_tile(tile, coord)

    def spawn_procent(self, tile: Tiles):
        '''
        Places tile at a procentage of open tiles
        '''
        spawn_amount: int = round(len(self.open_board_positions) * tile.procentual_spawn_rate)

        if (spawn_amount > 0):
            for coord in self.get_random_coords(spawn_amount):
                self.place_tile(tile, coord)