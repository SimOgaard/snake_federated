from math import exp
print(0.0001 + (1 - 0.0001) * exp(-1. * 200_000 / 40_000))