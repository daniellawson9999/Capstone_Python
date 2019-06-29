import gym
import numpy as np
from numpy import random

epsilon = 1
gamma = 1
alpha = .9

iterations = 10000
decay_rate = 1/iterations
test_iterations = 10000
wins = 0
draws = 0

env = gym.make('Blackjack-v0')
q_table = np.zeros([32,11,2,2]);


def convert_to_int(state):
    return (state[0],state[1],int(state[2]))

for i in range(iterations):
    state = convert_to_int(env.reset())
    
    for t in range(10):
        #print(state)
        if random.random() > epsilon:
            action = np.argmax(q_table[state])

        else:
            action = env.action_space.sample()
        #print(action)
        next_state, reward, done, info = env.step(action)
        next_state = convert_to_int(next_state)
        current_value = q_table[state][action]
        next_value= np.max(q_table[next_state])
        updated_value = (1-alpha) * current_value + alpha * (reward + gamma * next_value)
        q_table[state][action] = updated_value

        state = next_state

        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            epsilon -= decay_rate
            break
#evaluate
for i in range(test_iterations):
    state = convert_to_int(env.reset())
    for t in range(10):
        action = np.argmax(q_table[state])
        #action = env.action_space.sample()

        next_state, reward, done, info = env.step(action)
        state = convert_to_int(next_state)
        if done:
            if reward == 1:
                wins += 1
            elif reward == 0:
                draws += 1
            break
losses = test_iterations - wins - draws
print("{} win, {} draw, {} loss".format(wins/test_iterations, draws/test_iterations,losses/test_iterations))
