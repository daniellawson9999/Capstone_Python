import multienvironment
import environment
from enum import Enum,auto
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pylot as plt
import copy

#Used to define the overall use of the training
class Modes(Enum):
    TRAINING = auto()
    TESTING = auto()
    
#Type of environment to use. Eventually legacy, the environment class, will be fully replaced by multienvironment
class Env(Enum):
    LEGACY = auto()
    MULTI = auto()

#Numerical parameters for training
class Parameters(Enum):
    START_EPSILON = auto()
    GAMMA = auto()
    ALPHA = auto()
    EPISODES = auto()
    MAX_MOVES = auto()
    WIN_REWARD = auto()
    LOSS_REWARD = auto()
    MAX_MEMORY_SIZE = auto()
    BATCH_SIZE = auto()
    OPTIMIZER = auto()
    EPOCHS = auto()
    MIN_EPSILON = auto()
    #This should be a string, currently only 'linear' is supported
    EPSILON_DECAY = auto()
    #non-critical for training:
    TEST_EPOCHS = auto()
    TEST_MAX_MOVES = auto()

#based off deque example
class ReplayMemory():
    def __init__(self, size):
        self.memory = collections.deque(maxlen = size)
        
    def store(self, transition):
        #transition should be an array of s,a,l,r,s',t
        self.memory.append(transition)
        
    def sample(self,n):
        memory_size = len(self.memory)
        return np.asarray([self.memory[i] for i in 
                (np.random.choice(np.arange(memory_size),size=n, replace=False))])

    
#Enum that contains types of networks  
class Network(Enum):
    SA_TO_Q = auto()
    S_TO_QA = auto()
    
    def __init__(self, env_type, argument_variable = None, argument_file = None, 
                 training_variables = None, training_file = None, training_name=None, 
                 load_name = None, load_model = False, network_type, custom_network = None,
                 training_mode=Modes.TRAINING, use_tensorboard = True, print_training=True, require_all_params = False):
        
        self.env_type = env_type
        self.training_name = training_name
        self.training_mode = training_mode
        self.network_type = network_type
        self.test_epochs = test_epochs
        #check for valid environment settings
        if env_type == Env.LEGACY:
            self.env_import = environment
        elif env_type == Env.MULTI:
            self.env_import = multienvironment
        else:
            self.env_import = None
            raise Exception('invalid environment type, refer to the Env Enum for valid env types')
        
        #Load environment arguments
        if argument_file is not None:
            self.env_args = self.read_arguments(argument_file)
        elif argument_variable is not None:
            self.env_args = argument_variable
        else:
            self.env_args = None
            raise Exception('no environment argument parameters specified')
        
        #initialize the environment
        self.env = self.env_import.Environment(**self.env_args)   
        self.num_actions = self.env.action_space()
    
    
        #load training parameters and set as class fields
        if trainig_model is not Modes.TESTING: 
            #Load training arguments
            if training_file is not None:
                parameter_list = self.read_arguments(training_file)
            elif training_variables is not None:
                parameter_list = training_variables
            else:
                parameter_list = None
                raise Exception('no training argument parameters specified')
            
            #instead of a boolean which turns requirement on/off, there will be a list of params that are crucial for testing
            if require_all_params:
                for parameter in Parameters:
                    if parameter not in parameter_list:
                        raise Exception('parameter list is missing required parameters for training')
        
            #convert parameter list to class fields
            for parameter in parameter_list:
                #in case string based parameters are supported, this may not be needed
                setattr(self,parameter.name,parameter_list[parameter])
        
        #Set up the network, custom_network should pass an uncompiled model built with keras layers
        #either load a pre-trained model or create a new model
        if load_model:
            self.model = tf.keras.models.load_model('./models/' + load_name)
        else:
            #create a new model following one of the preset models, or create a new custom model
            if custom_network is not None:
                self.model = custom_network
            else:
                image_shape = np.shape(self.env.screenshot())
                num_actions = self.num_actions 
                if network_type == Network.SA_TO_Q:
                    
                    image_input = tf.keras.Input(shape=image_shape)
            
                    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation=tf.keras.activations.relu,strides=1)(image_input)
                    pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
                    drop1 = tf.keras.layers.Dropout(.25)(pooling1)
                    
                    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation=tf.keras.activations.relu)(drop1)
                    pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
                    drop2 = tf.keras.layers.Dropout(0.25)(pooling2)
                    
                    flat = tf.keras.layers.Flatten()(drop2)
                    conv_dense = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)(flat)
                    
                    action_input = tf.keras.Input(shape=(num_actions,))
                    action_dense = tf.keras.layers.Dense(num_actions**2, activation=tf.keras.activations.relu)(action_input)
                    
                    merged_dense = tf.keras.layers.concatenate([conv_dense, action_dense])
                    dense1 = tf.keras.layers.Dense(10, activation = tf.keras.activations.relu)(merged_dense)
                    output = tf.keras.layers.Dense(1,activation=tf.keras.activations.linear)(dense1)
                    
                    self.model = tf.keras.Model(inputs=[image_input,action_input], outputs = output)
                    
                elif network_type == Network.S_TO_QA:
                    #todo
                    raise Exception('todo')
                else:
                    raise Exception('invalid network type')
                #compile the model
        #self.model should now be defined, compile the model if training
        self.tensorboard = None
        self.replay_memory = None
        
        #other variables for training
        self.reward_lists = []
        self.epsilon = 1
        self.epsilon_decay_function = None
        
        if training_mode is not Modes.TESTING:
            #redo this logic eventually, support all tf optimizers 
            if self.OPTIMIZER == 'Adam':
                optimizer = tf.keras.optimizers.Adam
            elif self.OPTIMIZER == 'SGD':
                optimizer = tf.keras.optimizers.SGD
            else:
                raise Exception('optimizer not found')
            self.model.compile(loss=tf.keras.losses.mean_squared_error,optimizer = optimizer(lr = self.ALPHA))    
            #test custom directory thing
            
            #initialize epsilon decay function
            #right now, all functions should have 2 arg , epsilon and the total number of epochs
            self.epsilon_decay_function = self.linear_decay
        
            if use_tensorboard:
                self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}/{}'.format(training_name,time()),batch_size = batch_size,   write_grads=True,write_images=True)
                self.tensorboard.set_model(self.model)
                
    #function does everything at once
    def train_test_save(self, train_epochs = self.EPOCHS, test_epochs = self.TEST_EPOCHS, save_name = self.training_name):
        self.train(train_epochs)
        self.plot_rewards()
        self.test(test_epochs)
        self.save_all(save_name)
    #init object -> (init_replay_memeory -> train -> save -> test) or (func train_save_test)
    def init_replay_memory(self):
        self.replay_memory = ReplayMemory(self.MAX_MEMORY_SIZE)
        while len(self.replay_memory.memory):
            state = self.env.reset()
            done = False
            for i in range(self.MAX_MOVES):
                action = self.env.sample()
                #check this
                next_state, reward, done,_ = self.env.step(action)
                transition = [state, action, self.env.legal_actions(),reward,next_state,int(done)]
                state = next_state
                if len(self.replay_memory.memory) >= batch_size: break
    def linear_decay(self,epsilon, epochs):
        increment = 1/epochs
        new_epsilon = max(self.MIN_EPSILON, epsilon - increment)
        return new_epsilon
    
    def predict(self,state,legal_actions):
        actions = [0] * self.num_actions
        if self.network_type == Network.S_TO_QA:
            for i in range(self.num_actions):
                action = [0] * self.num_actions
                action[i] = 1
                #returns a 2d array such as [[8.0442]], we want to make an array of like [[234],[23432],[2343]]
                actions[i] = model.predict([np.expand_dims(state,0),np.expand_dims(action,0)])[0]
        elif self.network_type == Network.SA_TO_Q:
            #check this out
            actions = model.predict(np.expand_dims(state,0))
        else:    
            raise Exception('invalid network type')
        max_index = 0
        for i in range(len(actions)):
            if legal_actions[max_index] == 0 and legal_actions[i] == 1:
                max_index = i
            if actions[max_index][0] < actions[i][0] and legal_actions[i] == 1:
                max_index = i
        return max_index, actions[max_index][0]
    
    def train(self, epochs = self.EPOCHS):
        assert (self.training_mode == Modes.TRAINING), "the class was not initialized for training"
        #initialize the training replay memory, so that there is enough experience for the first training batch 
        self.init_replay_memory()
        self.epsilon = 1
        count = 0
        training_win = 0
        training_loss = 0
        for i in range(epochs):
            if print_training:
                print("starting iteration ", i)
            total_reward = 0
            if self.env_type == Env.MULTI:
                state = self.env.full_reset()
            else:
                state = self.env.reset()
            for m in range(self.MAX_MOVES):
                #select an action
                if np.random.random() > self.epsilon:
                    action,value = self.predict(state,self.env.legal_actions())
                else:
                    action = self.env.sample()
            #transition  
            next_state,reward, done, game_state = self.env.step(action)
            total_reward += reward
            #store transition
            transition = [state,action,self.env.legal_actions(),reward,next_state,int(done)]
            self.memory.store(transition)
            #sample batch of transitions from memory
            batch = self.memory.sample(self.BATCH_SIZE)
            #create lists to store the states, actions, targets from the batch that will be passed to the model to train
            states = list(np.zeros(batch_size))
            actions = list(np.zeros(batch_size))
            targets = list(np.zeros(batch_size))
            
            for j in range(self.BATCH_SIZE):
                
                #store the onehot encoded action
                a = batch[j,1]
                onehot = [0] * num_actions
                onehot[a] = 1
                actions[j] = onehot
                
                #store the state
                s = batch[j,0]
                states[j] = s
                
                #obtain the next state, legal_actions, reward, and terminal to determine the target
                n_s = batch[j,4]
                l = batch[j,2]
                r = batch[j,3]
                t = batch[j,5]
                #if a terminal state, the target q value is simply the reward,
                #otherwise use the standard equation
                if t:
                    targets[j] = r
                else:
                    index, value = predict(n_s,l)
                    targets[j] = r + gamma * value
            #convert back to np arrays for tensorflow
            states = np.asarray(states)
            actions = np.asarray(actions)
            if self.network_type == Network.SA_TO_Q:
                x = [states,actions]
            elif self.network_type ==  Network.S_TO_QA:
                x = states
            else:
                raise Exception('invalid network type in training')
            y = targets
            loss = model.train_on_batch(x = x, y = y)
            logs = {}
            logs['loss'] = loss
            tensorboard.on_epoch_end(count, logs)
            count += 1
            #update state
            state = next_state
            if done or m == max_moves - 1:
                if game_state == self.env.State.WIN:
                    training_win += 1
                if game_state == self.env.State.LOSS:
                    training_loss += 1
                print("total reward {} last iteration {} moves, total wins {}, total losses {}".format(total_reward,m+1,training_win,training_loss))
                print("epsilon", self.epsilon)
                #decrease epsilon following decay function
                epsilon = self.epsilon_decay_function(epsilon, epochs)
                self.reward_list.append(total_reward)
                break
        tensorboard.on_train_end(None)

    def test(self, epochs = self.TEST_EPOCHS):
        wins = 0
        losses = 0
        if self.env_type == Env.LEGACY:
            self.env.random_location = False
        for i in range(epochs):
            if self.env_type == Env.MULTI:
                state = self.env.full_reset()
            else:
                state = self.env.reset()
            for t in range(self.TEST_MAX_MOVES):
                action,value = self.predict(state,self.env.legal_actions())
                state,reward,done,game_state = self.env.step(action)
                if done:
                    if game_state == self.env.State.WIN:
                        training_win += 1
                    elif game_state == self.env.State.LOSS:
                        training_loss += 1 
                    break
        reached_max = epochs-wins-losses
        print("{} wins, {} losses, {} reached max".format(wins/epochs, losses/epochs,(reach_max)/epochs))
        return wins, losses, reached_max

    def plot_rewards(self):
        plt.plot(selfreward_list)
        plt.ylabel('total reward')
        plt.xlabel('episode')
        plt.show()    
         
    def dict_to_str(self,dictionary):
        new_dictionary = copy.deepcopy(dictionary)
        for key in dictionary:
            if type(dictionary[key]) is list:
                for i, item in enumerate(dictionary[key]):
                    if isinstance(item,Enum):
                        new_dictionary[key][i] = str(item)
            else:
                if isinstance(dictionary[key],Enum):
                        new_dictionary[key] = dictionary[key]
        return new_dictionary         
     #todo
    def save_all(self, save_name = self.training_name):
        #will do something
        save = None
    def save_model(self, save_name = self.training_name):
        self.model.save('./models/' + save_name)
    
    def write_arguments(self,file_name,args_dict):
        file = open('./model_env_arugments/' + file_name + '.txt')
        file.write(str(args_dict))
        file.close()
        
    def read_arguments(self, file_name):
        file = open('./model_env_arugments/' + file_name + '.txt')
        contents = file.read()
        dictionary = eval(contents)
        file.close()
        return dictionary
    
   

