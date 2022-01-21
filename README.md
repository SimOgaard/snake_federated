## ReinforcementFederatedLearning

Within this repository we combine Federated Learning with Reinforcement Learning on the game Snake with from scratch code to make updates on both sides as easy as possible to revise.

The `requirements.txt` file lists all Python libraries that this repository depends on. And can be install requirements using: ``` pip install -r requirements.txt ```

To use this repository from `google colab` or `DGX` we have helper files under `server_side_learning`.
The main files are located in the root directory; under the same directory this `README.md` is in with the naming convention: snake_*what_file_does*.py and with a description of the file under `if __name__ == "__main__":` in the .py file.

For example the python file ``` snake_terminal.py ``` has the description: ``` "Creates a environment with a controllable agent that is playable" ``` and can be run using this command: ``` python3 snake_terminal.py ```

## Results:
### Env:
Our snake environment was not meant to be efficient but easily itterated on so we could add more rules to our game and train/test our snakes in different environments. However we can still create ~800 boards per seconds with a size of 25x25 and achive ~15_000 steps per seconds from a random agent. And so our bottle neck is the NN forward and backward pass.

### RL:
With a board size of 20x20 and ~3h training time on Google Colab (Tesla K80 GPU) where our DQN agents state is a selected 7x7 kernel around its head of the environment and its coords to the closest pice of food we achive a minimum snake length of 41; average: 61.8; and max: 88

### RL_FED:
Training using FED average we achive

## Further research:
This can be either faults in code, optimization areas, pure UI or other NN/RL/FED algorithms to try out.

### Optimization:
* [This code](https://github.com/SimOgaard/snake_federated/blob/fc85c3bc567efef0b785bc600a8b191950b012ea/snake_env/snake_environment.py#L109) takes 45% of compute power.
* Traing takes a lot of cpu power (100%) and not a lot of gpu power (40%). Se if you cast tensor to right device and dtype. And if ReplayBuffer.sample can be improved uppon.
* Observation_food function requires cpu when not even using it (all_food_on_board: dict)

### UI:
* Fix ui for display in snake_terminal
* Print % amount of deaths that are from random action

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
* HamiltonianCycleAgent
* SupervisedLearningAgent
* BreadthFirstSearchAgent (hardcoded a* algorythm)

### FED:
* Get fedaverage to work over networks not just locally
* Add gosspip learning (P2P)

### Other
* Fedaverage fewer times