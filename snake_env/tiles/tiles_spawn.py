# Math Modules
from numpy.random import rand
from random import sample
from random import randint

# Repo imports
from snake_env.tiles.virtual_tiles import Tiles

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
        self.spawn_salt_pepper(tile, tile.salt_pepper_chance)
        self.spawn_procent(tile, tile.procentual_spawn_rate)
        self.spawn_epislon(tile)

    def spawn_number(self, tile: Tiles):
        '''
        Places a specified number of tiles on board at random places
        '''
        spawn_amount: int = randint(tile.spawn_amount[0], tile.spawn_amount[1])
        if (spawn_amount > 0):
            for coord in self.get_random_coords(spawn_amount):
                self.place_tile(tile, coord)

    def spawn_salt_pepper(self, tile: Tiles, salt_pepper_chance: float):
        '''
        Places specified tile on board at random places
        '''
        if (salt_pepper_chance > 0):
            for salt_pepper_val, coord in zip(self.get_salt_and_pepper(), list(self.open_board_positions)):
                if salt_pepper_val < salt_pepper_chance:
                    self.place_tile(tile, coord)

    def spawn_procent(self, tile: Tiles, procentual_spawn_rate: float):
        '''
        Places tile at a procentage of open tiles
        '''
        spawn_amount: int = round(len(self.open_board_positions) * procentual_spawn_rate)

        if (spawn_amount > 0):
            for coord in self.get_random_coords(spawn_amount):
                self.place_tile(tile, coord)

    def spawn_epislon(self, tile: Tiles):
        '''
        Places specified tile on board at random places given epislon function
        '''
        eps_val = tile.epsilon(self.run)
        if eps_val > 0:
            self.spawn_salt_pepper(tile, eps_val)
            # self.spawn_procent(tile, eps_val)
