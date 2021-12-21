# math functions
from numpy import array

# this repo imports
from snake_env.tiles.virtual_tiles import *

class AirTile(Tiles):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 0, reward: int = 0., occupy: bool = False) -> None:
        '''
        Initialize a air tile object.
        '''
        super(AirTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Does nothing
        '''
        return self.reward

class EdgeTile(Tiles):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 1, reward: int = -1., occupy: bool = True) -> None:
        '''
        Initialize a edge tile object.
        '''
        super(EdgeTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Kills snake
        '''
        snake.done = True
        return self.reward

class MineTile(Tiles):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 2, reward: int = -0.25, occupy: bool = True) -> None:
        '''
        Initialize a mine tile object.
        '''
        super(MineTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Removes tail from snake 
        Kills snake if snake body length is zero
        '''
        snake.remove_snake_tail()
        if (len(snake.snake_body) == 0):
            snake.done = True
            return -1 # return larger penalty because snake died
        return self.reward

class FoodTile(Tiles):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 3, reward: int = 1, occupy: bool = True) -> None:
        '''
        Initialize a food tile object.
        '''
        super(FoodTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> float:
        '''
        Adds new bodypart to snake
        '''
        # add snake part
        if ("old_snake_tail_pos" in kwargs):
            snake.place_new_snake_tail(kwargs["old_snake_tail_pos"])

        # place new food on board
        if (len(snake.board.open_board_positions) != 0):
            # get random coord viable for food placement
            random_coord: array = snake.board.random_open_tile_coord()
            snake.board.place_tile(FoodTile(), random_coord)
        return self.reward

class SnakeTile(Tiles):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 4, reward: int = -1., occupy: bool = True) -> float:
        '''
        Initialize a snake body tile object.
        '''
        super(SnakeTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs: dict) -> None:
        '''
        Kills snake
        '''
        snake.done = True
        return self.reward

class SnakeHeadTile(SnakeTile):
    '''
    Adds new bodypart to snake
    '''
    def __init__(self, visual: int = 5) -> None:
        '''
        Initialize a snake head tile object.
        '''
        super(SnakeHeadTile, self).__init__(visual = visual)
