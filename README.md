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

### WERID:
*       med air tile reward = -0.01 på environment med endast mines går snake in i mine
*       med air tile reward = 0.01 på environment med endast mines går snake inte in i mine
*       detta fungerar ej med federated learning

### TODO:
*       You should be able to have different tiles inits for different boards
*       Fix snake observation function near
*       Fix ui for display in snake_terminal
*       Store if it died from random choise or nah (print % amount after x episodes)

*       CNN DQN
*       PPO
*       AlphaGo route (test multiple actions)

*       Fruit that splits snake into two (currently not workking)
*       Fruit that detatches one bit of the tail and leaves it on board
*       Super fruit that gives more reward but gives you longer body

*       Random board structure not just cubic