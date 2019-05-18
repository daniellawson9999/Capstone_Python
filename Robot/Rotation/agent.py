import multienvironment
import environment
from enum import Enum,auto
import tensorflow as tf
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
import copy
from time import time


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
    MAX_MOVES = auto()
    WIN_REWARD = auto()
    LOSS_REWARD = auto()
    MAX_MEMORY_SIZE = auto()
    BATCH_SIZE = auto()
    OPTIMIZER = auto()
    EPOCHS = auto()
    MIN_EPSILON = auto()
    EPSILON_DECAY = auto()
    #non-critical for training:
    TEST_EPOCHS = auto()
    TEST_MAX_MOVES = auto()
    CONTINUOUS = auto()
    
class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()

class Decay(Enum):
    LINEAR = auto()
    
#Enum that contains types of networks  
class Network(Enum):
    SA_TO_Q = auto()
    S_TO_QA = auto()
    
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

    

    
class Agent():
    def __init__(self, env_type = Env.MULTI, env_dict = None, env_file_name = None, 
                 training_dict = None, training_file_name = None, env_dict_string = False, training_name=None, 
                 model_load_name = None, load_model = False, network_type = Network.SA_TO_Q, custom_network = None,
                 training_mode=Modes.TRAINING, use_tensorboard = True, print_training=True, require_all_params = False):
        
        self.env_type = env_type
        self.training_name = training_name
        self.training_mode = training_mode
        self.network_type = network_type
        self.print_training = print_training
        #check for valid environment settings
        if env_type == Env.LEGACY:
            self.env_import = environment
        elif env_type == Env.MULTI:
            self.env_import = multienvironment
        else:
            self.env_import = None
            raise Exception('invalid environment type, refer to the Env Enum for valid env types')
        
        
        #Load environment arguments
        if env_file_name is not None:
            self.env_args = self.read_dict('./args/environment/' + env_file_name + '.txt')
            if env_dict is not None:
                for key in env_dict:
                    self.env_args[key] = env_dict[key]
        elif env_dict is not None:
            if env_dict_string:
                env_dict = eval(env_dict)
            self.env_args = env_dict
        else:
            self.env_args = None
            raise Exception('no environment argument parameters specified')
        
        #initialize the environment
        self.env = self.env_import.Environment(**self.env_args)   
        self.num_actions = self.env.action_space()
    
    
        #load training parameters and set as class fields
        if training_mode is not Modes.TESTING: 
            #Load training arguments
            if training_file_name is not None:
                self.parameter_dict = self.read_dict('./args/training/' + training_file_name + '.txt')
                if training_dict is not None:
                    for key in training_dict:
                        self.parameter_dict[key] = training_dict[key]
            elif training_dict is not None:
                self.parameter_dict = training_dict
            else:
                self.parameter_dict = None
                raise Exception('no training argument parameters specified')
            
            #instead of a boolean which turns requirement on/off, there will be a list of params that are crucial for testing
            #if require_all_params:
              #  for parameter in Parameters:
                 #   if parameter not in parameter_dict:
                    #    raise Exception('parameter list is missing required parameters for training')
        
            #convert parameter list to class fields
            for parameter in self.parameter_dict:
                #in case string based parameters are supported, this may not be needed
                setattr(self,parameter.name,self.parameter_dict[parameter])
        
        #Set up the network, custom_network should pass an uncompiled model built with keras layers
        #either load a pre-trained model or create a new model
        if load_model:
            self.model = tf.keras.models.load_model('./models/' + model_load_name + '.h5')
        else:
            #create a new model following one of the preset models, or create a new custom model
            if custom_network is not None:
                self.model = custom_network
            else:
                image_shape = np.shape(self.env.screenshot())
                if self.env.frame_stacking:
                    image_shape += (self.env.stacker.stack_size,)
                print(image_shape)
                num_actions = self.num_actions 
                kernel_size = (5,5)
                if network_type == Network.SA_TO_Q:
                    
                    image_input = tf.keras.Input(shape=image_shape)
            
                    conv1 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=tf.keras.activations.relu,strides=1)(image_input)
                    pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
                    drop1 = tf.keras.layers.Dropout(.25)(pooling1)
                    
                    conv2 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=1, activation=tf.keras.activations.relu)(drop1)
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
                    image_input = tf.keras.Input(shape=image_shape)
            
                    conv1 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=tf.keras.activations.relu,strides=1)(image_input)
                    pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
                    drop1 = tf.keras.layers.Dropout(.25)(pooling1)
                    
                    conv2 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=1, activation=tf.keras.activations.relu)(drop1)
                    pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
                    drop2 = tf.keras.layers.Dropout(0.25)(pooling2)
                    
                    flat = tf.keras.layers.Flatten()(drop2)
                    conv_dense = tf.keras.layers.Dense(100, activation=tf.keras.activations.relu)(flat)
                    output = tf.keras.layers.Dense(num_actions,activation=tf.keras.activations.linear)(conv_dense)
                    
                    self.model = tf.keras.Model(inputs=image_input, outputs = output)
                else:
                    raise Exception('invalid network type')
                #compile the model
        #self.model should now be defined, compile the model if training
        self.tensorboard = None
        self.replay_memory = None
        
        #other variables for training
        self.reward_list = []
        self.epsilon = 1
        self.epsilon_decay_function = None
        
        if training_mode is not Modes.TESTING:
            #redo this logic eventually, support all tf optimizers
            if self.OPTIMIZER == Optimizer.ADAM:
                self.OPTIMIZER = tf.keras.optimizers.Adam
            elif self.OPTIMIZER == Optimizer.SGD:
                self.OPTIMIZER = tf.keras.optimizers.SGD
            if self.OPTIMIZER is None:
                self.OPTIMIZER = tf.keras.optimizers.Adam
            self.model.compile(loss=tf.keras.losses.mean_squared_error,optimizer = self.OPTIMIZER(lr = self.ALPHA))    
            #test custom directory thing
            
            #initialize epsilon decay function
            #right now, all functions should have 2 arg , epsilon and the total number of epochs
            if self.EPSILON_DECAY == Decay.LINEAR:
                self.epsilon_decay_function = self.linear_decay
            else:
                raise Exception('Decay function not found')
        
            if use_tensorboard:
                self.tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}/{}'.format(training_name,time()),batch_size = self.BATCH_SIZE,   write_grads=True,write_images=True)
                self.tensorboard.set_model(self.model)
                
    
    
    
    #function does everything at once
    def train_test_save(self, train_epochs = None, test_epochs = None, save_name = None):
        #set default parameters for this function, the none style is used b/c the defaulst reference object fields
        if train_epochs is None: train_epochs = self.EPOCHS
        if test_epochs is None: test_epochs = self.TEST_EPOCHS
        if save_name is None: save_name = self.training_name
        
        self.train(train_epochs)
        self.plot_rewards()
        self.test(test_epochs)
        self.save_all(save_name)
    #init object -> (init_replay_memeory -> train -> save -> test) or (func train_save_test)
    def init_replay_memory(self):
        self.replay_memory = ReplayMemory(self.MAX_MEMORY_SIZE)
        while len(self.replay_memory.memory) < self.BATCH_SIZE:
            state = self.env.reset()
            done = False
            for i in range(self.MAX_MOVES):
                action = self.env.sample()
                #check this
                next_state, reward, done,_ = self.env.step(action)
                transition = [state, action, self.env.legal_actions(),reward,next_state,int(done)]
                self.replay_memory.store(transition)
                state = next_state
                if len(self.replay_memory.memory) >= self.BATCH_SIZE: break
    def linear_decay(self,epsilon, epochs):
        increment = 1/epochs
        new_epsilon = max(self.MIN_EPSILON, epsilon - increment)
        return new_epsilon
    
    def predict(self,state,legal_actions):
        actions = [0] * self.num_actions
        if self.network_type == Network.SA_TO_Q:
            for i in range(self.num_actions):
                action = [0] * self.num_actions
                action[i] = 1
                #returns a 2d array such as [[8.0442]], we want to make an array of like [[234],[23432],[2343]]
                actions[i] = self.model.predict([np.expand_dims(state,0),np.expand_dims(action,0)])[0][0]
        elif self.network_type == Network.S_TO_QA:
            #check this out
            actions = self.model.predict(np.expand_dims(state,0))[0]
        else:    
            raise Exception('invalid network type')
        max_index = 0
        for i in range(len(actions)):
            if legal_actions[max_index] == 0 and legal_actions[i] == 1:
                max_index = i
            if actions[max_index] < actions[i] and legal_actions[i] == 1:
                max_index = i
        return max_index, actions[max_index]
    
   
    def train(self, epochs = None):
        if epochs is None:
            epochs = self.EPOCHS
        assert (self.training_mode == Modes.TRAINING), "the class was not initialized for training"
        #initialize the training replay memory, so that there is enough experience for the first training batch 
        self.init_replay_memory()
        self.epsilon = 1
        count = 0
        training_win = 0
        training_loss = 0
        for i in range(epochs):
            if self.print_training:
                print("starting iteration ", i)
            total_reward = 0
            if self.env_type == Env.MULTI:
                if self.CONTINUOUS:
                    state = self.env.reset()
                else:
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
                self.replay_memory.store(transition)
                #sample batch of transitions from memory
                batch = self.replay_memory.sample(self.BATCH_SIZE)
                #create lists to store the states, actions, targets from the batch that will be passed to the model to train
                states = list(np.zeros(self.BATCH_SIZE))
                actions = list(np.zeros(self.BATCH_SIZE))
                if self.network_type == Network.SA_TO_Q:
                    targets = list(np.zeros(self.BATCH_SIZE))
                elif self.network_type == Network.S_TO_QA:
                    targets = list(np.zeros((self.BATCH_SIZE,self.num_actions)))
                else:
                    raise Exception('invalid network type')
                
                
                for j in range(self.BATCH_SIZE):
                    
                    #store the onehot encoded action
                    a = batch[j,1]
                    onehot = [0] * self.num_actions
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
                    #the target is a little bit different depending on the style of the q network
                    #if SA-Q network is used, the target is simply a single value
                    #S_QA networks have multiple outputs, but we only want to adjust the network for the one
                    #action that this example is from, so we will make the predicted values
                    #for every other action the same as what the network currently predicts,
                    #so there is zero mean squared error on that action, meaning weights associated with 
                    #that action are not updated
                    if self.network_type == Network.SA_TO_Q:
                        #if a terminal state, the target q value is simply the reward,
                        #otherwise use the standard equation
                        if t:
                            targets[j] = r
                        else:
                            index, value = self.predict(n_s,l)
                            targets[j] = r + self.GAMMA * value
                    elif self.network_type == Network.S_TO_QA:
                        #init target array to for example j to be equal to the predicted
                        targets[j] = list(self.model.predict(np.expand_dims(states[j],0))[0])
                        #change the value for the desired action to be equal to b equation
                        if t:
                            targets[j][a] = r
                        else:
                            index, value = self.predict(n_s,l)
                            targets[j][a] = r + self.GAMMA * value
                       
                #convert back to np arrays for tensorflow
                states = np.asarray(states)
                actions = np.asarray(actions)
                targets = np.asarray(targets)
                if self.network_type == Network.SA_TO_Q:
                    x = [states,actions]
                elif self.network_type ==  Network.S_TO_QA:
                    x = states
                else:
                    raise Exception('invalid network type in training')
                y = targets
                loss = self.model.train_on_batch(x = x, y = y)
                logs = {}
                logs['loss'] = loss
                
                self.tensorboard.on_epoch_end(count, logs)
                count += 1
                #update state
                state = next_state
                if done or m == self.MAX_MOVES - 1:
                    if game_state == self.env_import.State.WIN:
                        training_win += 1
                    if game_state == self.env_import.State.LOSS:
                        training_loss += 1
                    if(self.print_training):
                        print(self.training_name + ": total reward {} last iteration {} moves, total wins {}, total losses {}".format(total_reward,m+1,training_win,training_loss))
                        print("epsilon", self.epsilon)
                    #decrease epsilon following decay function
                    self.epsilon = self.epsilon_decay_function(self.epsilon, epochs)
                    self.reward_list.append(total_reward)
                    break
        self.tensorboard.on_train_end(None)
        return training_win, training_loss

    def test(self, epochs = None):
        wins = 0
        losses = 0
        if epochs is None:
            epochs = self.TEST_EPOCHS
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
                    if game_state == self.env_import.State.WIN:
                        wins += 1
                    elif game_state == self.env_import.State.LOSS:
                        losses += 1 
                    break
        reached_max = epochs-wins-losses
        print(self.training_name + ": test results\n")
        print("{} wins, {} losses, {} reached max".format(wins/epochs, losses/epochs,(reached_max)/epochs))
        return wins, losses, reached_max

    def plot_rewards(self):
        fig = plt.figure()
        plt.plot(self.reward_list)
        fig.suptitle(self.training_name, fontsize=20)
        plt.ylabel('total reward')
        plt.xlabel('episode')
        plt.show()    
        return self.reward_list
         
    def dict_to_str(self,dictionary):
        string = str(dictionary)
        string = string.replace("<","")
        i = 0
        while(i < len(string)):
            if string[i] == ">":
                if string[i-4] == ":":
                    string = string[:i-4] + string[i+1:]    
                else:
                    string = string[:i-3] + string[i+1:]   
                i -= 3
            else:
                i += 1
        return string        
     #todo
    def save_all(self, save_name = None):
        if save_name is None: save_name = self.training_name
        self.save_parameter_dict(save_name)
        self.save_environment_dict(save_name)
        self.save_model(save_name)
        
    def save_environment_dict(self, save_name = None):
        path = './args/environment/' + save_name + '.txt'
        self.write_dict(path, self.env_args)
        
    def save_parameter_dict(self, save_name = None):
        path = './args/training/' + save_name + '.txt'
        self.write_dict(path, self.parameter_dict)

    def save_model(self, save_name = None):
        if save_name is None: save_name = self.training_name
        self.model.save('./models/' + save_name + '.h5')
    
    def write_dict(self,file_name,args_dict):
        file = open(file_name,"w+")
        file.write(self.dict_to_str(args_dict))
        file.close()
        
    def read_dict(self, file_name):
        Action = self.env_import.Action
        Reward = self.env_import.Reward
        Goal = self.env_import.Goal
        file = open(file_name,"r")
        contents = file.read()
        dictionary = eval(contents)
        file.close()
        return dictionary
    
   

