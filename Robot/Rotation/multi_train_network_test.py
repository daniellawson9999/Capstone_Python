import multiprocessing


def train(i):
    from multienvironment import Action, Reward, Goal
    from agent import Agent, Parameters, Decay, Modes, Env,Optimizer
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
    
    
    training_name_saq = "saq_test_stacked"
    training_name_sqa = "sqa_test_stacked"
    training_names  =[training_name_saq,training_name_sqa]
    env_dict_saq = {'width': 640, 'height': 480, 'mineral_scale': .5, 
            'camera_height': 3.5,'camera_tilt':0, 
            'actions':[Action.FORWARDS,Action.CW,Action.CCW], 
            'reward': Reward.RELATIVE_PROPORTIONAL,
            'decorations':True,'resize_scale':16, 'silver': (.8,.8,.8), 'random_colors':True,
            'random_lighting':True, 'silver_mineral_num':3, 'point_distance':9, 'stationary_scale':6,
            'normal_scale':2, 'stationary_win_count':5, 'shift_offset': 2,
            'goal': Goal.COLLISION, 'walls_terminal': True, 'close_all':False, 'figure_name': training_name_saq,
            'frame_stacking': True, 'stack_size': 3}
    
    env_dict_sqa = copy.deepcopy(env_dict_saq)
    env_dict_sqa['figure_name'] = training_name_sqa
    env_dicts = [env_dict_saq,env_dict_sqa]
    
    max_moves = 100
    epochs = 20
    training_dict_saq = {Parameters.START_EPSILON:1,Parameters.GAMMA:.95, Parameters.ALPHA:.001,
                 Parameters.EPOCHS: epochs, Parameters.MAX_MOVES:max_moves, Parameters.WIN_REWARD: 100,
                 Parameters.LOSS_REWARD: -100, Parameters.MAX_MEMORY_SIZE: max_moves*epochs,
                 Parameters.BATCH_SIZE: 16, Parameters.OPTIMIZER:  Optimizer.ADAM,
                 Parameters.MIN_EPSILON: .1, Parameters.TEST_EPOCHS: 20, 
                 Parameters.TEST_MAX_MOVES:max_moves, Parameters.EPSILON_DECAY: Decay.LINEAR, Parameters.CONTINUOUS: False}
    training_dict_sqa = copy.deepcopy(training_dict_saq)
    training_dicts = [training_dict_saq,training_dict_sqa]
    
    network_types = [Network.SA_TO_Q,Network.S_TO_QA]
    
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
    pool = multiprocessing.Pool(processes = 2)
    for i,results in enumerate(pool.imap(train,range(2))):
        print(results)
    
    
    