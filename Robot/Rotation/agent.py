import multienvironment
import environment
from enum import Enum,auto
import tensorflow as tf
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
import copy
import gym
from time import time
from network_builder import NetworkBuilder
from networks import Networks, Network


#Used to define the overall use of the training
class Modes(Enum):
    TRAINING = auto()
    TESTING = auto()
    
#Type of environment to use. Eventually legacy, the environment class, will be fully replaced by multienvironment
#Legacy and Multi are the Mayavi environments that the agent 
#was orginally built around
class Env(Enum):
    LEGACY = auto()
    MULTI = auto()
    GYM = auto()

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
    DOUBLE = auto()
    MAX_TAU = auto()
    TRACE_LENGTH = auto()
    PRE_TRAIN_EPOCHS = auto()
    UPDATE_FREQUENCY = auto()
    
    
class Optimizer(Enum):
    ADAM = auto()
    SGD = auto()

class Decay(Enum):
    LINEAR = auto()
        
#based off deque example
class ReplayMemory():
    def __init__(self,size):
        self.memory = collections.deque(maxlen = size)
  
    def store(self, transition):
        #transition should be an array of s,a,l,r,s',t
        self.memory.append(transition)
        
    def sample(self,n, random = True, trace_length = None):
        memory_size = len(self.memory)
        if random:
            return np.asarray([self.memory[i] for i in 
                (np.random.choice(np.arange(memory_size),size=n, replace=False))])
        else:
            assert(trace_length is not None), 'trace length not defined'
            sampled_memories = self.sample(n,random=True)
            sampled_traces = []
            for memory in sampled_memories:
                starting_point = np.random.randint(0,len(memory)+1-trace_length)
                sampled_traces.append(memory[starting_point:starting_point+trace_length])
            return np.asarray(sampled_traces)
        
    #returns the last *trace_length* states, returns the same state if there's not enough states
    #THIS SHOULD ONLY RETURN THE STATES, nOT THE ENTIRE TRANSITION
    def get_sequence(self,trace_length):
        #get the current sequence
        sequence = list(self.memory)[-trace_length]
        #which will not be long enough if there have been less than *trace_length* transitions
        while len(sequence) < trace_length:
            sequence.insert(0,sequence[0])
        #the memory is solely a list of sequences
        if len(np.shape(list(self.memory))) > 2:
            return np.asarray(sequence)
        else:
            #the memory contains a full transition
            array = []
            for state in sequence:
                #
                array.append(state[4])
            return np.asarray(array)

    
class Agent():
    def __init__(self, env_type = Env.MULTI, env_dict = None, env_file_name = None, 
                 training_dict = None, training_file_name = None, env_dict_string = False, training_name=None, 
                 model_load_name = None, load_model = False, network_type = Network.SA_TO_Q, custom_network = None,
                 training_mode=Modes.TRAINING, use_tensorboard = True, print_training=True, require_all_params = False, gym_env_name = None):
        
        self.env_type = env_type
        self.training_name = training_name
        self.training_mode = training_mode
        self.network_type = network_type
        self.print_training = print_training
        self.env_file_name = env_file_name
        self.env_dict = env_dict
        self.env_dict_string = env_dict_string
        self.gym_env_name = gym_env_name
        #check for valid environment settings
        if env_type == Env.LEGACY:
            self.env_import = environment
        elif env_type == Env.MULTI:
            self.env_import = multienvironment
        else:
            #A gym environment
            self.env_import = None
        
        #remove after testing
#        #Load environment arguments
#        if env_file_name is not None:
#            self.env_args = self.read_dict('./args/environment/' + env_file_name + '.txt')
#            if env_dict is not None:
#                for key in env_dict:
#                    self.env_args[key] = env_dict[key]
#        elif env_dict is not None:
#            if env_dict_string:
#                env_dict = eval(env_dict)
#            self.env_args = env_dict
#        else:
#            self.env_args = None
#        
#        #initialize the environment
#        self.env = self.env_import.Environment(**self.env_args)
        self.initialize_environment()
        #self.num_actions = self.env.action_space()
        self.num_actions = self.action_space()
    
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
            
            #initialize all training parameters as None, so those that are not used are still defined
            for parameter in Parameters:
                setattr(self,parameter.name,None)
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
            #image_shape = np.shape(self.env.screenshot())
            image_shape = self.observation_space()
            num_actions = self.num_actions 
            if custom_network is not None:
                
                argument_dict = {'image_shape':image_shape,'num_actions':num_actions}
                if network_type == Network.SM_TO_QA:
                    argument_dict['stack_size'] = self.env.stacker.stack_size
                if network_type == Network.SR_TO_QA:
                    argument_dict['trace_length'] = self.TRACE_LENGTH
                network_builder = NetworkBuilder(custom_network,network_type,argument_dict)
                self.model = network_builder.get_model()
            else:
                if self.env_type is not Env.GYM:
                    #fix this
                    if self.env.frame_stacking:
                        #a tupple
                        base_size = image_shape[0:2]
                        #a scalar
                        channels = image_shape[2]
                        stack_size = self.env.stacker.stack_size
                        #example stack- image dimensions: 30 x 40, stack size: 4, channels: 3
                        if self.env.concatenate:
                            #should be width by height by (channels * stack size)
                            #so 30 x 40 x 12
                            image_shape = base_size + (channels * stack_size,)
                        #else:
                            #hould be stack size by height  by width by channels
                            #so 4 x 30 x 40 x 3
                            #image_shape = (stack_size,) + base
                    #print(image_shape)
                kernel_size = (5,5)
                
                #default models for each network type
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
                   
                elif network_type == Network.SM_TO_QA:
                    #concat should be false
                    stack_size = self.env.stacker.stack_size
                    input_layer_list = []
                    dense_layer_list = []
                    for i in range(stack_size):
                       
                        image_input = tf.keras.Input(shape=image_shape)
                        
                        conv1 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, activation=tf.keras.activations.relu,strides=1)(image_input)
                        pooling1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
                        drop1 = tf.keras.layers.Dropout(.25)(pooling1)
                    
                        conv2 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, strides=1, activation=tf.keras.activations.relu)(drop1)
                        pooling2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
                        drop2 = tf.keras.layers.Dropout(0.25)(pooling2)
                            
                        
                        flat = tf.keras.layers.Flatten()(drop2)
                        
                        #add to layer lists
                        input_layer_list.append(image_input)
                        dense_layer_list.append(flat)
                        
                    merged_dense = tf.keras.layers.concatenate(dense_layer_list)
                    dense1 = tf.keras.layers.Dense(100,activation=tf.keras.activations.relu)(merged_dense)
                    dense2 = tf.keras.layers.Dense(100,activation=tf.keras.activations.relu)(dense1)
                    output = tf.keras.layers.Dense(num_actions,activation=tf.keras.activations.linear)(dense2)
                    
                    self.model = tf.keras.Model(inputs = input_layer_list, outputs = output)
                else:
                    raise Exception('invalid network type or no default model for network type')
            
        
        #other variables for training
        self.target_model = None
        self.tensorboard = None
        self.replay_memory = None
        self.reward_list = []
        self.epsilon = 1
        self.epsilon_decay_function = None
        
        #self.model should now be defined, compile the model if training
        if training_mode is not Modes.TESTING:
            #copy the model if using double q learning
            if self.DOUBLE:
                self.target_model = tf.keras.models.clone_model(self.model)
        
            #redo this logic eventually, support all tf optimizers
            if self.OPTIMIZER == Optimizer.ADAM:
                self.OPTIMIZER = tf.keras.optimizers.Adam
            elif self.OPTIMIZER == Optimizer.SGD:
                self.OPTIMIZER = tf.keras.optimizers.SGD
            if self.OPTIMIZER is None:
                self.OPTIMIZER = tf.keras.optimizers.Adam
            self.model.compile(loss=tf.keras.losses.mean_squared_error,optimizer = self.OPTIMIZER(lr = self.ALPHA))
            if self.DOUBLE:
                self.update_target()
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
            if self.UPDATE_FREQUENCY is None:
                self.UPDATE_FREQUENCY = 1
    
    
    
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
        #possibly disbale max memory size for SR_TO_QA networks
        self.replay_memory = ReplayMemory(self.MAX_MEMORY_SIZE)
        #Recurrent networks require different replay memory logic
        #Their replay memories must be initialized to store a larger history of an entire epochs
        #This allows a sequence of states with length TRACE_LENGTH to be sampled from each episode
        #In this case, the main replay memory contains a list of individual replay memory objects for each epoch
        if self.network_type == Network.SR_TO_QA:
            while len(self.replay_memory.memory) < self.PRE_TRAIN_LENGTH:
                epoch_memory = ReplayMemory(size = None)
                state = self.reset_environment()
                done = False
                for i in range(self.MAX_MOVES):
                    action = self.sample_action()
                    next_state, reward,done,_ = self.env.step(action)
                    transition = [state,action,self.legal_actions(),reward,next_state,int(done)]
                    epoch_memory.store(transition)
                    state = next_state
                    if done: break
                self.replay_memory.store(epoch_memory)
        else:
            while len(self.replay_memory.memory) < self.BATCH_SIZE:
                state = self.reset_environment()
                done = False
                for i in range(self.MAX_MOVES):
                    action = self.sample_action()
                    #check this
                    next_state, reward, done,_ = self.env.step(action)
                    transition = [state, action, self.legal_actions(),reward,next_state,int(done)]
                    self.replay_memory.store(transition)
                    state = next_state
                    if len(self.replay_memory.memory) >= self.BATCH_SIZE: break
            
    def linear_decay(self,epsilon, epochs):
        increment = 1/epochs
        new_epsilon = max(self.MIN_EPSILON, epsilon - increment)
        return new_epsilon
    
    def update_target(self):
        if self.target_model is None:
            raise Exception("the target network has not been initialized")
        self.target_model.set_weights(self.model.get_weights())
        
    #returns the maximum q value corresponding action
    #given an input state, legal actions
    #defaults to self.model, can pass another model such as self.target_model
    #can pass another action using index
    def predict(self,state,legal_actions,model=None, index = None):
        if model is None:
            model = self.model
        actions = [0] * self.num_actions
        if self.network_type == Network.SA_TO_Q:
            for i in range(self.num_actions):
                action = [0] * self.num_actions
                action[i] = 1
                #returns a 2d array such as [[8.0442]], we want to make an array of like [[234],[23432],[2343]]
                actions[i] = model.predict([np.expand_dims(state,0),np.expand_dims(action,0)])[0][0]
        elif self.network_type == Network.S_TO_QA or self.network_type == Network.SR_TO_QA:
            #check this out
            actions = model.predict(np.expand_dims(state,0))[0]
        elif self.network_type == Network.SM_TO_QA:
            actions = list(model.predict([frame for frame in np.expand_dims(state,1)]))[0]
            
        else:    
            raise Exception('invalid network type')
        max_index = 0
        for i in range(len(actions)):
            if legal_actions[max_index] == 0 and legal_actions[i] == 1:
                max_index = i
            if actions[max_index] < actions[i] and legal_actions[i] == 1:
                max_index = i
        return_index = index
        if index is None:
            return_index = max_index
        return return_index, actions[return_index]
    
   
    def train(self, epochs = None):
        #number of steps taken
        t = 0
        if epochs is None:
            epochs = self.EPOCHS
        assert (self.training_mode == Modes.TRAINING), "the class was not initialized for training"
        #initialize the training replay memory, so that there is enough experience for the first training batch 
        
        #the agent takes random actions until the replay memory has enough experience
        self.init_replay_memory()
        
        self.epsilon = 1
        
        #number of batches trained
        count = 0
        
        #number of wins and losses during training
        training_win = 0
        training_loss = 0
        
        for i in range(epochs):
            if self.print_training:
                print("starting iteration ", i)
                
            total_reward = 0
            
        
            state = self.reset_environment()
            
            epoch_memory = None
            
            if self.network_type == Network.SR_TO_QA:
                epoch_memory = ReplayMemory(size = None)
                #store first state temporally to act on, will be deleted and replaced by the transition
                epoch_memory.store(state)
                
            for m in range(self.MAX_MOVES):
                #select an action
                if np.random.random() > self.epsilon:
                    if self.network_type == Network.SR_TO_QA:
                        s = epoch_memory.get_sequence(self.TRACE_LENGTH)
                        #remove first state to replace w/ transition
                        if len(epoch_memory) == 1:
                            epoch_memory.memory.pop()
                    else:
                        s = state
                    action,value = self.predict(s,self.legal_actions())
                else:
                    action = self.sample_action()
                #transition  
                next_state,reward, done, game_state = self.env.step(action)
                total_reward += reward
                #store transition
                transition = [state,action,self.legal_actions(),reward,next_state,int(done)]
                
                if self.network_type == Network.SR_TO_QA:
                    self.replay_memory.store(transition)
                else:
                    epoch_memory.store(transition)
                
                if t % self.UPDATE_FREQUENCE == 0:
                    #sample batch of transitions from memory
                    if self.network_type == Network.SR_TO_QA:
                        #if using a SR_TO_QA network, the batch contains entire epoch memories, however we instead want a trace
                        batch = self.replay_memory.sample(self.BATCH_SIZE, random=False,trace_length=self.TRACE_LENGTH)
                        #32x8x40x30x3    
                    else:
                        batch = self.replay_memory.sample(self.BATCH_SIZE)
                        
                    #create lists to store the states, actions, targets from the batch that will be passed to the model to train
                    states = list(np.zeros(self.BATCH_SIZE))
                    actions = list(np.zeros(self.BATCH_SIZE))
                    if self.network_type == Network.SA_TO_Q:
                        targets = list(np.zeros(self.BATCH_SIZE)) #32x1
                    elif self.network_type == Network.S_TO_QA or self.network_type == Network.SM_TO_QA or self.network_type == Network.SR_TO_QA:
                        targets = list(np.zeros((self.BATCH_SIZE,self.num_actions))) #32xnjum_actions
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
                        #if using double q learning the target the q value of the max action calculated by the main q-network
                        #def predict(self,state,legal_actions,model=None, index = None):
                        if self.network_type == Network.SA_TO_Q:
                            #if a terminal state, the target q value is simply the reward,
                            #otherwise use the standard equation
                            if t:
                                targets[j] = r
                            else:
                                action, value = self.predict(n_s,l)
                                if self.DOUBLE:
                                    value = self.predict(n_s,l,self.target_model,index=action)
                                targets[j] = r + self.GAMMA * value
                        elif self.network_type == Network.S_TO_QA or Network.SM_TO_QA or Network.SR_TO_QA:
                            #init target array to for example j to be equal to the predicted
                           
                            
                            if self.network_type == Network.S_TO_QA or Network.SR_TO_QA:
                                targets[j] = list(self.model.predict(np.expand_dims(states[j],0))[0])
                            else:
                                targets[j] = list(self.model.predict([frame for frame in np.expand_dims(states[j],1)]))[0]
    
                            #change the value for the desired action to be equal to b equation
                            if t:
                                targets[j][a] = r
                            else:
                                action, value = self.predict(n_s,l)
                                if self.DOUBLE:
                                    value = self.predict(n_s,l,self.target_model,index=action)
                                targets[j][a] = r + self.GAMMA * value
                            
                           
                    #convert back to np arrays for tensorflow
                    states = np.asarray(states)
                    actions = np.asarray(actions)
                    targets = np.asarray(targets)
                    if self.network_type == Network.SA_TO_Q:
                        x = [states,actions]
                    elif self.network_type ==  Network.S_TO_QA or Network.SR_TO_QA:
                        x = states
                    elif self.network_type == Network.SM_TO_QA:
                        #targets[j] = list(self.model.predict([frame for frame in np.expand_dims(states[j],1)]))[0]
                        #x = np.expand_dims(states,2)\
                        output = [[] for i in range(self.BATCH_SIZE)]
                        for i in range(self.BATCH_SIZE):
                            for j in range(self.env.stack_size):
                                output[j].append(states[i][j])
                        x= output
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
                
                #update target network if using double q learning and tau is greater than max_tau
                if self.DOUBLE:
                    if t % self.MAX_TAU == 0:
                        self.update_target()
                    
                t += 1
                
                if done or m == self.MAX_MOVES - 1:
                    #custom env also returns game state
                    if self.env_type is not Env.GYM:
                        if game_state == self.env_import.State.WIN:
                            training_win += 1
                        if game_state == self.env_import.State.LOSS:
                            training_loss += 1
                    if(self.print_training):
                        print(self.training_name + ": total reward {} last iteration {} moves, total wins {}, total losses {}".format(total_reward,m+1,training_win,training_loss))
                        print("epsilon", self.epsilon)
                    
                    #decrease epsilon following decay function
                    self.epsilon = self.epsilon_decay_function(self.epsilon, epochs)
                    
                    #add to the award list for printing
                    self.reward_list.append(total_reward)
                    
                    #add epoch memory to the replay memory
                    if self.network_type == Network.SR_TO_QA: 
                        self.replay_memory.store(epoch_memory)
                        
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
            
            epoch_memory = None
            if self.network_type == Network.SR_TO_QA:
                epoch_memory = ReplayMemory(size = None)
                
            state = self.reset_environment()
            
            epoch_memory.store(state)
            
            for t in range(self.TEST_MAX_MOVES):
                if self.network_type == Network.SR_TO_QA:
                    s = epoch_memory.get_sequence(self.TRACE_LENGTH)
                else:
                    s = state
                action,value = self.predict(s,self.env.legal_actions())
                state,reward,done,game_state = self.env.step(action)
                if self.network_type == Network.SR_TO_QA:
                    epoch_memory.store(state)
                if done:
                    if self.env_type is not Env.GYM:
                        if game_state == self.env_import.State.WIN:
                            wins += 1
                        elif game_state == self.env_import.State.LOSS:
                            losses += 1 
                    break
        reached_max = 0 
        if self.env_type is not Env.GYM:
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
    
    #environment functions
    
    def initialize_environment(self):
        if self.env_type is not Env.GYM:
                #Load environment arguments
            if self.env_file_name is not None:
                self.env_args = self.read_dict('./args/environment/' + self.env_file_name + '.txt')
                if self.env_dict is not None:
                    for key in self.env_dict:
                        self.env_args[key] = self.env_dict[key]
            elif self.env_dict is not None:
                if self.env_dict_string:
                    self.env_dict = eval(self.env_dict)
                self.env_args = self.env_dict
            else:
                self.env_args = None
            self.env = self.env_import.Environment(**self.env_args)
        else:
            self.env = gym.make(self.gym_env_name)
     
    def action_space(self):
        if self.env_type is not Env.GYM:
            return self.env.action_space()
        else:
            return self.env.action_space.n
    
    def observation_space(self):
        if self.env_type is not Env.GYM:
            return np.shape(self.env.screenshot())
        else:
            return self.env.observation_space.shape
     
    #resets the environnment and returns the state
    def reset_environment(self):
        if self.env_type is not Env.GYM:
            if self.env_type == Env.MULTI:
                    if self.CONTINUOUS:
                        return self.env.reset()
                    else:
                        return self.env.full_reset()
            else:
                return self.env.reset()
        else:
            return self.env.reset()
    
    #randomly sampels an action
    def sample_action(self):
        if self.env_type is not Env.GYM:
            return self.env.sample()
        else:
            return self.env.action_space.sample()
    #returns the legal actions,gym envs do not have this implementation
    def legal_actions(self):
        if self.env_type is not Env.GYM:
            return self.env.legal_actions()
        else:
            return [1] * self.action_space()
    
     
    #parameter/environment saving functions
    #environment saving not supported for Gym envs yet 
    def save_all(self, save_name = None):
        if save_name is None: save_name = self.training_name
        self.save_parameter_dict(save_name)
        if self.env_type is not Env.GYM:
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
        # = self.env_import.Action
        #Reward = self.env_import.Reward
        #Goal = self.env_import.Goal
        file = open(file_name,"r")
        contents = file.read()
        dictionary = eval(contents)
        file.close()
        return dictionary
    
   

