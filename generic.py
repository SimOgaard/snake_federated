from random import randint

def better_rand(x: int, y: int):
    '''
    Workaround this stupid fucking code snippet: assert x != y in random.randrange
    '''
    return randint(x, y)

    # if (x != y):
    # return x
