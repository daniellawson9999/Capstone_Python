import multiprocessing

def train(i):
    from multienvironment import Action, Reward, Goal
    from agent import Agent, Parameters, Decay, Modes, Env, Network, Optimizer
    import tensorflow as tf
    import matplotlib
    import copy
    matplotlib.interactive(True)
    
    #configuration fix from https://sefiks.com/2019/03/20/tips-and-tricks-for-gpu-and-multiprocessing-in-tensorflow/
    core_config = tf.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.Session(config=core_config)
    tf.keras.backend.set_session(session)
    
    
    training_name_sqa = "sqa_stacked_1"
    training_name_smqa = "smqa_stacked_1_2"
    training_name_sqa_2 = "sqa_stacked_2"
    training_name_smqa_2 = "smqa_stacked_2"
    training_names  =[training_name_sqa,training_name_smqa,training_name_sqa_2,training_name_smqa_2]
    env_dict_sqa = {'width': 640, 'height': 480, 'mineral_scale': .5, 
            'camera_height': 3.5,'camera_tilt':0, 
            'actions':[Action.FORWARDS,Action.CW,Action.CCW], 
            'reward': Reward.RELATIVE_PROPORTIONAL,
            'decorations':True,'resize_scale':16, 'silver': (.8,.8,.8), 'random_colors':True,
            'random_lighting':True, 'silver_mineral_num':3, 'point_distance':9, 'stationary_scale':6,
            'normal_scale':2, 'stationary_win_count':5, 'shift_offset': 2,
            'goal': Goal.COLLISION, 'walls_terminal': True, 'close_all':False, 'figure_name': training_name_sqa,
            'frame_stacking': True, 'stack_size': 4, 'concatenate': True, 'penalize_turning':True}
    
    env_dict_sqa_2 = copy.deepcopy(env_dict_sqa)
    env_dict_sqa_2['figure_name'] = training_name_sqa_2
    
    env_dict_smqa = copy.deepcopy(env_dict_sqa)
    env_dict_smqa['figure_name'] = training_name_smqa
    env_dict_smqa['concatenate'] = False

    
    env_dict_smqa_2 = copy.deepcopy(env_dict_smqa)
    env_dict_smqa_2['figure_name'] = training_name_smqa_2
    
    env_dicts = [env_dict_sqa,env_dict_smqa,env_dict_sqa_2,env_dict_smqa_2]
    
    max_moves = 150
    epochs = 1500
    training_dict_1 = {Parameters.START_EPSILON:1,Parameters.GAMMA:.95, Parameters.ALPHA:.001,
                 Parameters.EPOCHS: epochs, Parameters.MAX_MOVES:max_moves, Parameters.WIN_REWARD: 100,
                 Parameters.LOSS_REWARD: -100, Parameters.MAX_MEMORY_SIZE: max_moves*epochs,
                 Parameters.BATCH_SIZE:32, Parameters.OPTIMIZER:  Optimizer.ADAM,
                 Parameters.MIN_EPSILON: .01, Parameters.TEST_EPOCHS: 20, 
                 Parameters.TEST_MAX_MOVES:max_moves, Parameters.EPSILON_DECAY: Decay.LINEAR, 
                 Parameters.CONTINUOUS: False}
    training_dict_2 = copy.deepcopy(training_dict_1)
    training_dict_2[Parameters.ALPHA] = .0003
    training_dicts = [training_dict_1,training_dict_1,training_dict_2,training_dict_2]
    
    network_types = [Network.S_TO_QA,Network.SM_TO_QA,Network.S_TO_QA,Network.SM_TO_QA]
    
    agent = Agent(env_type = Env.MULTI, env_dict = env_dicts[i], 
              training_dict = training_dicts[i],  network_type = network_types[i], training_name = training_names[i])
    
    training_results = agent.train()
    
    agent.save_all()

    reward_list = agent.plot_rewards()
    
    test_results = agent.test()
    
    name = agent.training_name
    
    session.close()
    tf.keras.backend.clear_session()
    
    return (name,training_results,test_results)
    
if __name__ == "__main__":
    #pool_size = 4
    #pool = multiprocessing.Pool(processes = pool_size)
    #for i,results in enumerate(pool.imap(train,range(pool_size))):
    #    print(results)
    p = multiprocessing.Process(target=train, args=(1,))
    p.start()
    p.join()
   
    


