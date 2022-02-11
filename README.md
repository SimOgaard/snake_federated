## ReinforcementFederatedLearning

Within this repository we combine Federated Learning with Reinforcement Learning on the game [Snake](https://g.co/kgs/PgHC17) with from scratch code to make updates on both sides as easy as possible to revise.

The `requirements.txt` file lists all Python libraries that this repository depends on. And can be installed using: ``` pip install -r requirements.txt ```. Other than that we have the file ``` snake_terminal.py ``` that will require you to be on windows os and install ``` pip install getch ```

To use this repository from `google colab` or `DGX` we have helper files under `server_side_learning`.
The main files are located in the root directory; under the same directory this `README.md` with a naming convention that describes what it does and with a lengthier description of the file under `if __name__ == "__main__":` in the .py file. All main files are reasearch files made for the [presentation](./doc/ReinforcementFederatedLearningWithNotes.pptx). 

For example the python file ``` snake_terminal.py ``` has the description: ``` "Creates a environment with a controllable agent that is playable" ``` and can be run using this command: ``` python3 snake_terminal.py ```

## Results:
### Env:
Our snake environment was not meant to be efficient but easily itterated on so we could add more rules to our game and train/test our snakes in different environments. However we can still create ~800 boards per seconds with a size of 25x25 and achive ~15_000 steps per seconds from a random agent. And so our bottle neck is the NN forward and backward pass.

### RL:
With a board size of 20x20 and ~3h training time on Google Colab (Tesla K80 GPU) where our DQN agents state is a selected 7x7 kernel around its head of the environment and its coords to the closest pice of food we achive a minimum snake length of 41; average: 61.8; and max: 88.

### RL_FED:
Training using FED average we achive

![](/doc/gif/fednone_food/replay_fed_none_food_5x5+4_episode_100000.gif) ![](/doc/gif/fednone_food/replay_fed_none_food_5x5+4_episode_180000.gif)

![](/doc/gif/fed_food/replay_fed_food_5x5+4_episode_80000.gif) ![](/doc/gif/fed_food/replay_fed_food_5x5+4_episode_140000.gif)

![](/doc/gif/fednone_food_red_mines/replay_fed_none_food_red_mine_5x5+4_episode_100000.gif) ![](/doc/gif/fednone_food_red_mines/replay_fed_none_food_red_mine_5x5+4_episode_180000.gif)

![](/doc/gif/fed_food_blue_mine/replay_fed_food_blue_mine_5x5+4_episode_80000.gif) ![](/doc/gif/fed_food_blue_mine/replay_fed_food_blue_mine_5x5+4_episode_140000.gif)

![](/doc/gif/fed_food_blue_mine/new_environment/replay_fed_food_red_mine_5x5+4_episode_15000.gif) ![](/doc/gif/fed_food_blue_mine/new_environment/replay_fed_food_red_blue_mine_5x5+4_episode_15000.gif)

![](/doc/gif/fed_food_blue_and_red_mines/replay_fed_food_blue_and_red_mine_5x5+4_episode_80000.gif) ![](/doc/gif/fed_food_blue_and_red_mines/replay_fed_food_blue_and_red_mine_5x5+4_episode_140000.gif)

![](/doc/gif/fedlate_food_blue_and_red_mines/replay_fed_late_food_blue_and_red_mine_5x5+4_episode_200000.gif) ![](/doc/gif/fedlate_food_blue_and_red_mines/replay_fed_late_food_blue_and_red_mine_5x5+4_episode_220000.gif)

### Optimization:
* [Filling open_board_position dict takes 45% of ```def __restart__(self)``` compute](https://github.com/aidotse/ReinforcementFederatedLearning/blob/6421ea8ea9ea321bdb04c543274d38234135adf5/snake_env/snake_environment.py#L110)
* Traing takes a lot of cpu power (100%) and not a lot of gpu power (40%). Se if you cast tensor to right device and dtype. And if ReplayBuffer.sample can be improved uppon.
* Observation_food function requires cpu when not even using it (all_food_on_board: dict)

### UI:
* Print % amount of deaths that are from random action
* Make matplotlib replay work with snake_terminal.py

### Env:
* Food that splits snake into two (currently not working)
* Food that detatches one bit of the tail and leaves it on board
* Super food that gives more reward but gives you longer body
* Random board structure not just cubic

### NN:
* CNN DQN
* PPO
* AlphaGo route (test multiple actions)
* PolicyGradient
* AdvantageActorCriticAgent

### FED:
* Fedaverage between episodes
* Get fedaverage to work over networks not just locally
* Add gosspip learning (P2P)

### General:
* Have agents compete and learn from each other in the same environment​
* Train benchmark: different epsilon greedy policies for a network of federated snakes to see if we can accelerate training​
