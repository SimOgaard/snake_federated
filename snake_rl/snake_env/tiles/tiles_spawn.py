# Math Modules
from numpy.random import rand
from random import sample

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
