# math functions
from numpy import array

# this repo imports
from snake_env.tiles.virtual_tiles import *

class AirTile(Tiles): 
    def __init__(self, visual: int = 0, reward: int = -0.01, occupy: bool = False) -> None:
        super(AirTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs) -> None:
        pass

class EdgeTile(Tiles):
    def __init__(self, visual: int = 1, reward: int = -1., occupy: bool = True) -> None:
        super(EdgeTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs) -> None:
        snake.done = True

class MineTile(Tiles):
    def __init__(self, visual: int = 2, reward: int = -0.25, occupy: bool = True) -> None:
        super(MineTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs) -> None:
        snake.remove_snake_tail()
        if (len(snake.snake_body) == 0):
            snake.done = True

class FoodTile(Tiles):
    def __init__(self, visual: int = 3, reward: int = 1, occupy: bool = True) -> None:
        super(FoodTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs) -> None:
        # add snake part
        if ("old_snake_tail_pos" in kwargs):
            snake.place_new_snake_tail(kwargs["old_snake_tail_pos"])

        # place new food on board
        if (len(snake.board.open_board_positions) != 0):
            # get random coord viable for food placement
            random_coord: array = snake.board.random_open_tile_coord()
            snake.board.place_tile(FoodTile(), random_coord)

class SnakeTile(Tiles):
    def __init__(self, visual: int = 4, reward: int = -1., occupy: bool = True) -> None:
        super(SnakeTile, self).__init__(visual, reward, occupy)

    def on_hit(self, snake, **kwargs) -> None:
        snake.done = True

class SnakeHeadTile(SnakeTile):
    def __init__(self, visual: int = 5) -> None:
        super(SnakeHeadTile, self).__init__(visual = visual)
