#this class implements trainining using the agent class
from multienvironment import Action, Reward, Goal
from agent import Agent, Parameters, Decay, Modes, Env, Network, Optimizer
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.interactive(True)

training_name = "saq_test_01_2"
env_file_name = training_name
model_load_name = training_name
network_type = Network.SA_TO_Q
training_file_name = training_name
#create agent using dictionaries
agent = Agent(env_type = Env.MULTI, env_file_name = env_file_name, 
              network_type = network_type,  training_name = training_name, training_file_name = training_file_name, model_load_name = model_load_name,load_model=True, training_mode = Modes.TESTING)

agent.TEST_MAX_MOVES = 50
wins, losses, reached_max = agent.test(epochs=100)

