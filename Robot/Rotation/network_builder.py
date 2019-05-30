import tensorflow as tf
from networks import Networks, Network
from tensorflow.keras import layers
#used in the agent class to build a non-default model
#a training class should not create a NetworkBuilder if multiprocessing is being used
#all implementations pass a Netowrks enum to the agent class
#which then creates and uses the NetworkBuilder
#if you would like to implemnet your own network,
#the easiest way is to add your own network function
#and then that function name to the function_dictionary, so the function
#can be used in a training class
class NetworkBuilder():
    def __init__(self, network_name, network_type, argument_dictionary):
        self.network_name = network_name
        self.argument_dictionary = argument_dictionary
        self.function_dictionary = {
                Networks.DOOM_CNN_SM: self.DOOM_CNN_SM
        }
        self.function_type_dictionary = {
                Networks.DOOM_CNN_SM: Network.SM_TO_QA
        }
        #make sure that the network has a type
        assert(self.function_type_dictionary[network_name]==network_type),"mismatched network type"
    def get_model(self):
        return self.function_dictionary[self.network_name](**self.argument_dictionary)
    def DOOM_CNN_SM(self,image_shape,num_actions,stack_size):
        input_layer_list = []
        dense_layer_list = []
        for i in range(stack_size):
           
            image_input = tf.keras.Input(shape=image_shape)
            
            conv1 = layers.Conv2D(32, kernel_size=(8,8), activation=tf.keras.activations.relu,strides=(4,4),padding='valid')(image_input)
            norm1 = layers.BatchNormalization()(conv1)
            
        
            conv2 = layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), activation=tf.keras.activations.relu,padding='valid')(norm1)
            norm2 = layers.BatchNormalization()(conv2)
            
            #conv3 = layers.Conv2D(128, kernel_size=(4,4), strides=(2,2), activation=tf.keras.activations.relu,padding='valid')(norm2)
            #norm3 = layers.BatchNormalization()(conv3)
            
            flat = tf.keras.layers.Flatten()(norm2)
            
            #add to layer lists
            input_layer_list.append(image_input)
            dense_layer_list.append(flat)
            
        merged_dense = tf.keras.layers.concatenate(dense_layer_list)
        dense1 = layers.Dense(256,activation=tf.keras.activations.relu)(merged_dense)
        dense2 = layers.Dense(128,activation=tf.keras.activations.relu)(dense1)
        output = layers.Dense(num_actions,activation=tf.keras.activations.linear)(dense2)
        
        model = tf.keras.Model(inputs = input_layer_list, outputs = output)
        return model