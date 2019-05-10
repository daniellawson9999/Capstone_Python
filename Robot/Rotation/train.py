#this class implements trainining using the agent class
from multienvironment import Action, Reward
from agent import Agent, Parameters, Decay, Modes, Env, Network, Optimizer
import tensorflow as tf
import numpy as np

#dictionary containing arguments to initialize the environment
env_dict = {'width': 640, 'height': 480, 'mineral_scale': .5, 
            'camera_height': 3.5,'camera_tilt':0, 
            'actions':[Action.FORWARDS,Action.CW,Action.CCW,Action.STAY], 
            'reward': Reward.RELATIVE_PROPORTIONAL,
            'decorations':True,'resize_scale':16, 'silver': (.8,.8,.8), 'random_colors':True,
            'random_lighting':True, 'silver_mineral_num':3, 'point_distance':9, 'stationary_scale':6,
            'normal_scale':2, 'stationary_win_count':5, 'shift_offset': 2}
#dictionary containing arguments for training settings

training_dict = {Parameters.START_EPSILON:1,Parameters.GAMMA:.95, Parameters.ALPHA:.001,
                 Parameters.EPOCHS: 5, Parameters.MAX_MOVES:50, Parameters.WIN_REWARD: 100,
                 Parameters.LOSS_REWARD: -100, Parameters.MAX_MEMORY_SIZE: 10 * 150,
                 Parameters.BATCH_SIZE: 16, Parameters.OPTIMIZER:  Optimizer.ADAM,
                 Parameters.MIN_EPSILON: .1, Parameters.TEST_EPOCHS: 3, 
                 Parameters.TEST_MAX_MOVES:10, Parameters.EPSILON_DECAY: Decay.LINEAR}

#create agent using dictionaries
agent = Agent(env_type = Env.MULTI, env_dict = env_dict, 
              training_dict = training_dict,  network_type = Network.S_TO_QA, training_name = 'test_training')
  
#Or load from a previously stored    
#agent = Agent(env_type = Env.MULTI, env_file_name = "test_training", 
#              training_file_name = "test_training", load_model = True, model_load_name = "test_training", 
#             network_type = Network.SA_TO_Q, training_name = "test_training")


#train, test, and save the agent. If you want to do alll of this, just run agent.train_test_save()

agent.train()

agent.plot_rewards()

agent.test()

#saves the environment and training arguments as well as the model
agent.save_all()