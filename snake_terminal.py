# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import pretty_print, display

if __name__ == "__main__":
    '''
    Creates a environment with a controllable agent that is playable
    '''
    board_dim: int = 9

    player_snake: ControllableAgent = ControllableAgent(init_snake_lengths=array([2, 2]))

    board: Board = Board(
        min_board_shape         = array([9, 9]),
        max_board_shape         = array([9, 9]),
        replay_interval         = 0,
        snakes                  = [player_snake],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile()
        },
    )

    while True:
        board.__restart__()

        while board.is_alive():
            state = observation_full(board=board)
            pretty_print(state, array([board_dim+2,board_dim+2]))
            print(observation_food(snake=player_snake))
            if (not player_snake.done):
                action, is_random = player_snake.act()
                reward: float = player_snake.move(action, is_random)

        input("You died with final length of {}...".format(len(player_snake.snake_body)))