# Repo imports
from snake_env.snake_environment import *
from snake_env.snake_agents.agents import *

from composed_functions.dqn_helper import load_checkpoint, save_checkpoint, load_checkpoint_to_snake

# Math modules
from numpy import mean as numpy_mean

if __name__ == "__main__":
    '''
    Trains a DQN-agent
    '''

    episode_amount: int = 100_000
    save_every: int = 1_000
    board_dim: int = 20
    state_size: int = 7
    model_id: str = "{}x{}+{}".format(state_size, state_size, 4)
    model_path: str = 'models/checkpoint{}.pth'.format(model_id)

    dqn_snake: DQNAgent = DQNAgent(
        state_size          = state_size**2+4,
        action_size         = 4,
        init_snake_lengths  = array([2, 2]),
        seed                = 1337,
        batch_size          = 128,
        gamma               = 0.999,
        epsilon             = Epsilon(1, 0., 25_000),
        learning_rate       = 1e-4,
        tau                 = 1e-3,
        update_every        = 32,
        buffer_size         = 1_000_000
    )

    board: Board = Board(
        min_board_shape         = array([board_dim, board_dim]),
        max_board_shape         = array([board_dim, board_dim]),
        replay_interval         = 0,
        snakes                  = [dqn_snake],
        tiles_populated         = {
            "air_tile": AirTile(),
            "wall_tile": WallTile(),
            "food_tile": FoodTile(epsilon=Epsilon(1,0,10_000))
        },
    )

    scores_window = deque(maxlen=100) # last 100 scores

    checkpoint = load_checkpoint(model_path)
    load_checkpoint_to_snake(dqn_snake, checkpoint)

    while board.run < episode_amount:
        
        board.__restart__() # restart board
        state: FloatTensor = observation_cat(observation_near(board=board, snake=dqn_snake, kernel=state_size), observation_food(dqn_snake)) # save init state

        while board.is_alive(): # check if snakes are alive

            action, is_random = dqn_snake.act(state) # choose an action for given snake
            reward: float = dqn_snake.move(action, is_random)

            action = LongTensor([action]) # take the agents action that leed to that reward and state
            reward = FloatTensor([reward]) # take the reward that the agent stored

            next_state: FloatTensor = observation_cat(observation_near(board=board, snake=dqn_snake, kernel=state_size), observation_food(dqn_snake)) # observe what steps taken lead to

            dqn_snake.step(state, action, reward, next_state, dqn_snake.done) # signal step to snake

            state = next_state # set old state to the next state

        scores_window.append(len(dqn_snake.snake_body)) # save the most recent score
        print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), dqn_snake.epsilon(dqn_snake.board.run)), end="")
        
        if board.run != 0:
            if board.run % 100 == 0:
                print('\rEpisode {}\tAverage Score {:.3f}\tRandom act chance {:.6f}'.format(board.run, numpy_mean(scores_window), dqn_snake.epsilon(dqn_snake.board.run)))
            if board.run % save_every == 0:
                save_checkpoint(dqn_snake, model_path)