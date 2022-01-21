# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

# Generic imports
import time

if __name__ == "__main__":
    '''
    Creates specified environment and counts;
        achived APS (actions per second) of specified agent
        achived board inits (how many boards can be instanciated per second)
    '''
    random_snake: RandomAgent = RandomAgent(init_snake_lengths=array([2, 10]))

    board: Board = Board(
        min_board_shape         = array([500, 500]),
        max_board_shape         = array([500, 500]),
        replay_interval         = 0,
        snakes                  = [random_snake],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile()
        },
    )

    board_reset_amount: int = 0
    board_reset_time = 1e-5

    board_step_amount: int = 0
    board_step_time = 1e-5

    while True:
        start_time: float = time.time()
        board.__restart__()
        board_reset_amount += 1
        board_reset_time += time.time() - start_time

        start_time: float = time.time()
        while board.is_alive():
            action, is_random = random_snake.act()
            reward: float = random_snake.move(action, is_random)

            board_step_amount += 1
        board_step_time += time.time() - start_time

        print('\rBoards Per Second: {:.5f} Steps Per Second: {:.5f}'.format(board_reset_amount / board_reset_time, board_step_amount / board_step_time), end="")