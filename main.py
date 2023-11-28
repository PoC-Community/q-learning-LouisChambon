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

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> np.ndarray:
    """
    This function updates the Q-table based on the Q-learning formula and returns the updated Q-table.
    """

    old_q_value = q_table[state, action]
    max_new_q_value = np.max(q_table[newState, :])
    new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_RATE * max_new_q_value - old_q_value)
    q_table[state, action] = new_q_value
    
    return q_table

qTable = init_q_table(5, 4)

print("Q-Table:\n" + str(qTable))

sample_q_table = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]
])

sample_state = 1
sample_action = 2
sample_reward = -1
sample_newState = 3

updated_q_table = q_function(sample_q_table, sample_state, sample_action, sample_reward, sample_newState)

print("\nUpdated Q-Table: " + str(updated_q_table))

assert(np.mean(qTable) == 0)

def main():
    return 0

if (__name__ == "__main__"):
    exitcode = main()
