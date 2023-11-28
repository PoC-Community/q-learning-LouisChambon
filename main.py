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

LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    """
    This function updates the Q-table based on the Q-learning formula and returns the updated Q-value.
    """

    old_q_value = q_table[state, action]
    max_new_q_value = np.max(q_table[newState, :])
    new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_RATE * max_new_q_value - old_q_value)
    q_table[state, action] = new_q_value
    
    return new_q_value


q_table = init_q_table(5,4)

q_table[0, 1] = q_function(q_table, state=0, action=1, reward=-1, newState=3)

print("Q-Table after action:\n" + str(q_table))

assert(q_table[0, 1] == -LEARNING_RATE), f"The Q function is incorrect: the value of qTable[0, 1] should be -{LEARNING_RATE}"

def main():
    return 0

if (__name__ == "__main__"):
    exitcode = main()
