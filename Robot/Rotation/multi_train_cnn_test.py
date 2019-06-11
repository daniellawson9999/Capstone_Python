import multiprocessing
from networks import Networks

def train(i):
    from multienvironment import Action, Reward, Goal
    from agent import Agent, Parameters, Decay, Modes, Env, Optimizer
    from networks import Network
    import tensorflow as tf
    import matplotlib
    import copy
    matplotlib.interactive(True)
    
    #configuration fix from https://sefiks.com/2019/03/20/tips-and-tricks-for-gpu-and-multiprocessing-in-tensorflow/
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.Session(config=core_config)
    tf.keras.backend.set_session(session)
    
    
    training_name_default = "default_cnn_1"
    training_name_doom= "doom_cnn_1"
    training_name_default_2= "default_cnn_2"
    training_name_doom_2 = "doom_cnn_2"
    training_names  =[training_name_default,training_name_doom,training_name_default_2,training_name_doom_2]
    env_dict_default = {'width': 640, 'height': 480, 'mineral_scale': .5, 
            'camera_height': 3.5,'camera_tilt':0, 
            'actions':[Action.FORWARDS,Action.CW,Action.CCW], 
            'reward': Reward.RELATIVE_PROPORTIONAL,
            'decorations':True,'resize_scale':16, 'silver': (.8,.8,.8), 'random_colors':True,
            'random_lighting':True, 'silver_mineral_num':3, 'point_distance':9, 'stationary_scale':6,
            'normal_scale':2, 'stationary_win_count':5, 'shift_offset': 2,
            'goal': Goal.COLLISION, 'walls_terminal': True, 'close_all':False, 'figure_name': training_name_default,
            'frame_stacking': True, 'stack_size': 3, 'concatenate': False, 'penalize_turning':True}
    
    env_dict_default_2 = copy.deepcopy(env_dict_default)
    env_dict_default_2['figure_name'] = training_name_default_2
    env_dict_default_2['reward'] = Reward.TERMINAL
    
    env_dict_doom = copy.deepcopy(env_dict_default)
    env_dict_doom['figure_name'] = training_name_doom
    
    env_dict_doom_2 = copy.deepcopy(env_dict_default_2)
    env_dict_doom_2['figure_name'] = training_name_doom_2
    
    env_dicts = [env_dict_default,env_dict_doom,env_dict_default_2,env_dict_doom_2]
    
    max_moves = 150
    epochs = 2000
    training_dict_1 = {Parameters.START_EPSILON:1,Parameters.GAMMA:.95, Parameters.ALPHA:.0005,
                 Parameters.EPOCHS: epochs, Parameters.MAX_MOVES:max_moves, Parameters.WIN_REWARD: 100,
                 Parameters.LOSS_REWARD: -100, Parameters.MAX_MEMORY_SIZE: max_moves*epochs,
                 Parameters.BATCH_SIZE:32, Parameters.OPTIMIZER:  Optimizer.ADAM,
                 Parameters.MIN_EPSILON: .01, Parameters.TEST_EPOCHS: 20, 
                 Parameters.TEST_MAX_MOVES:max_moves, Parameters.EPSILON_DECAY: Decay.LINEAR, 
                 Parameters.CONTINUOUS: False}

    training_dicts = [training_dict_1,training_dict_1,training_dict_1,training_dict_1]
    
    custom_networks = [None,Networks.DOOM_CNN_SM,None,Networks.DOOM_CNN_SM]
    
    agent = Agent(env_type = Env.MULTI, env_dict = env_dicts[i], custom_network=custom_networks[i],
              training_dict = training_dicts[i],  network_type = Network.SM_TO_QA, training_name = training_names[i])
    
    training_results = agent.train()
    
    agent.save_all()

    reward_list = agent.plot_rewards()
    
    test_results = agent.test()
    
    name = agent.training_name
    
    session.close()
    tf.keras.backend.clear_session()
    
    return (name,training_results,test_results)
    
if __name__ == "__main__":
    pool_size = 4
    pool = multiprocessing.Pool(processes = pool_size)
    for i,results in enumerate(pool.imap(train,range(pool_size))):
        print(results)
    #p = multiprocessing.Process(target=train, args=(1,))
    #p.start()
    #p.join()
   
    


