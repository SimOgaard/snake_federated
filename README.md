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
python3 snake_federated.py
```

To get this repo woring on google colab:
```
!git clone https://github.com/SimOgaard/snake_federated
%cd /content/snake_federated
```

### TODO:
###     Fruit that splits snake into two (currently not workking)
###     Fruit that detatches one bit of the tail and leaves it on board
###     Super fruit that gives more reward but gives you longer body
###     Random board structure
###
###     print if it died from random choise or nah
###     dgx ag100 kort gpu är inte snabb lamao