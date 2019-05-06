import multienvironment
import environment
from enum import Enum,auto
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pylot as plt

#Used to define the overall use of the training
class Modes(Enum):
    TRAINING = auto()
    TESTING = auto()
    TRAINING_TESTING = auto()
    
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


    
#Enum that contains types of networks  
class Network(Enum):
    SA_TO_Q = auto()
    S_TO_QA = auto()
    
def __init__(self, env_type, argument_variable = None, argument_file = None, 
             training_variables = None, training_file = None, training_name=None, 
             save_all_settings=True, load_name = None, load_model = False, network_type, custom_network = None,
             training_mode=Modes.TRAINING_TESTING):
    
    self.training_name = training_name
    self.save_all_settings = save_all_settings
    self.training_mode = training_mode
    self.network_type = network_type

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
        
        #make sure all listed parameters are defined, this is done seperately incase this can be overrided 
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
    if training_mode is not Modes.TESTING:
        #redo this logic eventually, support all tf optimizers 
        if self.OPTIMIZER == 'Adam':
            optimizer = tf.keras.optimizers.Adam
        elif self.OPTIMIZER == 'SGD':
            optimizer = tf.keras.optimizers.SGD
        else:
            raise Exception('optimizer not found')
        self.model.compile(loss=tf.keras.losses.mean_squared_error,optimizer = optimizer(lr = self.ALPHA))
    
def predict(self,state,legal_actions):
    actions = [0] * self.num_actions
    if self.network_type == Network.S_TO_QA:
        for i in range(self.num_actions):
            action = [0] * self.num_actions
            action[i] = 1
            actions[i] = model.predict([np.expand_dims(state,0),np.expand_dims(action,0)])[0]
    elif self.network_type == Network.SA_TO_Q:
        #check this out
        actions = model.predict(np.expand_dims(state,0))[0]
    else:    
        raise Exception('invalid network type')
    max_index = 0
    for i in range(len(actions)):
        if legal_actions[max_index] == 0 and legal_actions[i] == 1:
            max_index = i
        if actions[max_index][0] < actions[i][0] and legal_actions[i] == 1:
            max_index = i
    return max_index, actions[max_index][0]

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