#!/usr/bin/env python3

import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)

def init_q_table(x: int, y: int) -> np.ndarray:
    """
    This function must return a 2D matrix containing only zeros for values.
    """
    return np.zeros((x, y))

LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    try:
        old_q_value = q_table[state, action]
        max_new_q_value = np.max(q_table[newState, :])
        new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_RATE * max_new_q_value - old_q_value)
        q_table[state, action] = new_q_value
    except Exception as e:
        print("Error in q_function:")
        print(f"state: {state}, action: {action}, reward: {reward}, newState: {newState}")
        raise e
    
    return new_q_value

def random_action(env):
    return random.randint(0, env.action_space.n - 1)

def best_action(q_table: np.ndarray, state: int) -> int:
    """
    Write a function which finds the best action for the given state.

    It should return its index.
    """
    return np.argmax(q_table[state, :])

def game_loop(env: gym.Env, q_table: np.ndarray, state: int, action: int) -> tuple:
    print(env.step(action))
    next_state, reward, done, _, _ = env.step(action)
    print(state, action)
    q_table = q_function(q_table, state, action, reward, next_state)
    return q_table, next_state, done, reward


def choose_action(epsilon: float, q_table: np.ndarray, state: int, env: gym.Env) -> int:
    if random.random() > epsilon:
        return best_action(q_table, state)
    else:
        return random_action(env)

EPOCH = 20000

q_table = init_q_table(env.observation_space.n, env.action_space.n)

for i in range(EPOCH):
    state, info = env.reset()
    while True:
        action = random_action(env)
        q_table, state, done, reward = game_loop(env, q_table, state, action)
        if done:
            break

for states in q_table:
    for actions in states:
        if actions == max(states):
            print("\033[4m", end="")
        else:
            print("\033[0m", end="")
        if actions > 0:
            print("\033[92m", end="")
        else:
            print("\033[00m", end="")
        print(round(actions, 3), end="\t")
    print()

epsilon = 1.0
for i in range(10000):
    epsilon = max(epsilon - 0.0001, 0)
    state, info = env.reset()
    while True:
        action = choose_action(epsilon, q_table, state, env)
        q_table, state, done, reward = game_loop(env, q_table, state, action)
        if done:
            break

# Testing the AI
wins = 0.0
for i in range(100):
    state, info = env.reset()
    while True:
        action = choose_action(0, q_table, state, env)
        _, state, done, reward = game_loop(env, q_table, state, action)
        if done:
            if reward > 0:
                wins += 1
            break

print(f"{round(wins / (i+1) * 100, 2)}% winrate")
print(wins)

plt.imshow(env.render())

env.close()
def main():
    return 0

if (__name__ == "__main__"):
    exitcode = main()
