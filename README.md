## ReinforcementFederatedLearning

Within this repository we combine Federated Learning with Reinforcement Learning on the game [Snake](https://g.co/kgs/PgHC17) with from scratch code to make updates on both sides as easy as possible to revise.

The `requirements.txt` file lists all Python libraries that this repository depends on. And can be installed using: ``` pip install -r requirements.txt ```. Other than that we have some optional files like ``` snake_terminal.py ``` that will require you to be on windows os and install ``` pip install getch ``` and ``` snake_testing_save.py ``` that will require [imagemagick](https://imagemagick.org/script/download.php).

To use this repository from `google colab` or `DGX` we have helper files under `server_side_learning`.
The main files are located in the root directory; under the same directory this `README.md` with a naming convention that describes what it does and with a lengthier description of the file under `if __name__ == "__main__":` in the .py file. All main files are reasearch files made for the [presentation](./doc/ReinforcementFederatedLearningWithNotes.pptx). 

For example the python file ``` snake_terminal.py ``` has the description: ``` "Creates a environment with a controllable agent that is playable" ``` and can be run using this command: ``` python3 snake_terminal.py ```

## Results:
### Env:
Our snake environment was not meant to be efficient but easily itterated on so we could add more rules to our game and train/test our snakes in different environments. However we can still create ~800 boards per seconds with a size of 25x25 and achive ~15_000 steps per seconds from a random agent. And so our bottle neck is the NN forward and backward pass.

### Models
All models were trained on a board size of 22x22 on Google Colab (Tesla k80 GPU) where the snake sees a 5x5 kernel around it's head and the direction of the food compared to it's head. All models except for fedlate were trained for ~7 hours while late federated aggregated two fully trained models and trained for ~4 hours extra. The following is described more in the [presentation](./doc/ReinforcementFederatedLearningWithNotes.pptx).

The score is taken from the latest version of the model and may change according to random chance

### No federated only food

Min: 37; Average: 44.0; Max: 69; Stuck: 0

![](/doc/gif/fednone_food/replay_fed_none_food_5x5+4_episode_100000.gif) ![](/doc/gif/fednone_food/replay_fed_none_food_5x5+4_episode_180000.gif)

### Federated only food

Min: 36; Average: 42.6; Max: 59; Stuck: 0

![](/doc/gif/fed_food/replay_fed_food_5x5+4_episode_80000.gif) ![](/doc/gif/fed_food/replay_fed_food_5x5+4_episode_140000.gif)

### No federated food and red mines

Min: 25; Average: 41.6; Max: 60; Stuck: 0

![](/doc/gif/fednone_food_red_mines/replay_fed_none_food_red_mine_5x5+4_episode_100000.gif) ![](/doc/gif/fednone_food_red_mines/replay_fed_none_food_red_mine_5x5+4_episode_180000.gif)

### Federated food and blue mines

Min: 23; Average: 39.2; Max: 86; Stuck: 0

![](/doc/gif/fed_food_blue_mine/replay_fed_food_blue_mine_5x5+4_episode_80000.gif) ![](/doc/gif/fed_food_blue_mine/replay_fed_food_blue_mine_5x5+4_episode_140000.gif)

### Federated food and blue mines in environments with red mines

![](/doc/gif/fed_food_blue_mine/new_environment/replay_fed_food_red_mine_5x5+4_episode_15000.gif) ![](/doc/gif/fed_food_blue_mine/new_environment/replay_fed_food_red_blue_mine_5x5+4_episode_15000.gif)

### Federated food and blue+red mines

Min: 1; Average: 27.8; Max: 62; Stuck: 1

![](/doc/gif/fed_food_blue_and_red_mines/replay_fed_food_blue_and_red_mine_5x5+4_episode_80000.gif) ![](/doc/gif/fed_food_blue_and_red_mines/replay_fed_food_blue_and_red_mine_5x5+4_episode_140000.gif)

### Late federated food and blue+red mines

Min: 35; Average: 48.2; Max: 55; Stuck: 0

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
