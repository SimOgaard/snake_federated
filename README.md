The `requirements.txt` file lists all Python libraries that this repository depends on.

Install requirements using:
```
pip install -r requirements.txt
```

To test playing a game of snake run:
```
python3 snake_terminal.py
```

To train a DQN snake agent run:
```
python3 snake_training.py
```

To train two DQN snake agents on different environments with federated learning run:
```
python3 snake_federated_transfer_learning.py
```

To get this repo woring on google colab:
```
!git clone https://github.com/SimOgaard/snake_federated
%cd /content/snake_federated
```

## Just straight up weird:
* med air tile reward = -0.01 på environment med endast mines går snake in i mine
* med air tile reward = 0.01 på environment med endast mines går snake inte in i mine
* detta fungerar ej med federated learning

## TODO:
### Optimization:
* [this bitch](https://github.com/SimOgaard/snake_federated/blob/fc85c3bc567efef0b785bc600a8b191950b012ea/snake_env/snake_environment.py#L109) takes 99% of compute power
* Traing takes a lot of cpu power (100%) and not a lot of gpu power (35%). Se if you cast tensor to right device and dtype. And if ReplayBuffer.sample can be improved uppon.

### UI:
* Fix ui for display in snake_terminal
* Store if it died from random choise or nah (print % amount after x episodes)

### NN:
* CNN DQN
* PPO
* AlphaGo route (test multiple actions)

### Env:
* Fruit that splits snake into two (currently not working)
* Fruit that detatches one bit of the tail and leaves it on board
* Super fruit that gives more reward but gives you longer body
* Random board structure not just cubic