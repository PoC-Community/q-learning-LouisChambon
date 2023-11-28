#!/usr/bin/env python3

import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# Initializing an empty table

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return np.zeros((x, y))

qTable = init_q_table(5, 4)

print("Q-Table:\n" + str(qTable))

assert(np.mean(qTable) == 0)

def main():
    return 0

if (__name__ == "__main__"):
    exitcode = main()
