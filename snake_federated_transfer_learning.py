# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.ui_helper import pretty_print, display, display_run
from composed_functions.dqn_helper import dqn, load_checkpoint, load_checkpoint_to_snake, save_checkpoint
from federated_learning.federated_average import agregate

if __name__ == "__main__":
    '''
    Trains two seperated DQN-agents in two seperate environments with different rules
    Does a fedaverage between the two agents to show that two agents in seperate environments can transfer what they have learned to eachother
    '''

    episode_amount: int = 10_000
    env_episode_amount: int = 100
    save_every: int = 5
    board_dim: int = 20
    state_size: int = 5
    model_id: str = "{}x{}".format(state_size, state_size)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    # Snake and its environment with only mines
    dqn_snake_mine: DQNAgent = DQNAgent(
        state_size          = state_size**2, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0., 100_000),
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )
    board_mine: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_mine],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "mine_tile": MineTile()
        },
    )

    # Snake and its environment with food
    dqn_snake_food: DQNAgent = DQNAgent(
        state_size          = state_size**2, #(board_dim + 2)**2,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 69,
        batch_size          = 512,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0., 100_000),
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )
    board_food: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_food],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile()
        },
    )

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake_mine, checkpoint)
    load_checkpoint_to_snake(dqn_snake_food, checkpoint)

    for i in range(episode_amount):
        # train each snake seperatly for env_episode_amount episodes
        mine_median: float = dqn(board_mine, dqn_snake_mine, env_episode_amount, lambda: observation_near(board=board_mine, snake=dqn_snake_mine, kernel=array([state_size, state_size])))
        food_median: float = dqn(board_food, dqn_snake_food, env_episode_amount, lambda: observation_near(board=board_mine, snake=dqn_snake_mine, kernel=array([state_size, state_size])))

        print('\rEpisode {}\tAverage Scores ({:.3f}, {:.3f})\tRandom act chance {:.6f}'.format(i * env_episode_amount, mine_median, food_median, dqn_snake_food.epsilon(dqn_snake_food.board.run)))

        # do a fedaverage between them
        agregate([dqn_snake_mine, dqn_snake_food])

        # Save their model
        if i % save_every == 0:
            save_checkpoint(dqn_snake_mine, model_path)

    # test the snake that hadnt seen any foods only mines on food and mine board and see how it reacts
    board_combined: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake_mine],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(),
            "mine_tile": MineTile()
        },
    )

    # display 10 runs
    for _ in range(10):
        display_run(board_combined, dqn_snake_mine, array([state_size, state_size]), pretty_print, lambda: observation_near(board=board_mine, snake=dqn_snake_mine, kernel=array([state_size, state_size])))