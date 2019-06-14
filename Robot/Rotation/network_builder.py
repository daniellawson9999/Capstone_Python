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
                Networks.DOOM_CNN_SM: self.DOOM_CNN_SM,
                Networks.DUELING_SM: self.DUELING_SM,
                Networks.DUELING_S: self.DUELING_S,
                Networks.DUELING_LSTM: self.DUELING_LSTM_SM
        }
        self.function_type_dictionary = {
                Networks.DOOM_CNN_SM: Network.SM_TO_QA,
                Networks.DUELING_SM: Network.SM_TO_QA,
                Networks.DUELING_S: Network.S_TO_QA,
                Networks.DUELING_LSTM: Network.SR_TO_QA
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
            
            flat = layers.Flatten()(norm2)
            
            #add to layer lists
            input_layer_list.append(image_input)
            dense_layer_list.append(flat)
            
        merged_dense = tf.keras.layers.concatenate(dense_layer_list)
        dense1 = layers.Dense(256,activation=tf.keras.activations.relu)(merged_dense)
        dense2 = layers.Dense(128,activation=tf.keras.activations.relu)(dense1)
        output = layers.Dense(num_actions,activation=tf.keras.activations.linear)(dense2)
        
        model = tf.keras.Model(inputs = input_layer_list, outputs = output)
        return model
    def DUELING_S(self,image_shape,num_actions):
            #from dueling example
            image_input = tf.keras.Input(shape=image_shape)
            conv1 = layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), activation=tf.keras.activations.relu,padding='valid')(image_input)
            conv2 = layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), activation=tf.keras.activations.relu,padding='valid')(conv1)
            conv3 = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation=tf.keras.activations.relu,padding='valid')(conv2)
            flatten = layers.Flatten()(conv3)
            
            advantage_fc = layers.Dense(512, activation='relu')(flatten)
            advantage = layers.Dense(num_actions)(advantage_fc)
            advantage = layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),
                               output_shape=(self.action_size,))(advantage)
    
            value_fc = layers.Dense(512, activation='relu')(flatten)
            value =  layers.Dense(1)(value_fc)
            value = layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),
                           output_shape=(num_actions,))(value)
    
            q_values = layers.merge([value, advantage], mode='sum')
            model = tf.keras.Model(inputs=image_input, outputs=q_values)
            return model
    def DUELING_SM(self,image_shape,num_actions,stack_size):
        input_layer_list = []
        dense_layer_list = []
        for i in range(stack_size):
           
            image_input = tf.keras.Input(shape=image_shape)
            conv1 = layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), activation=tf.keras.activations.relu,padding='valid')(image_input)
            conv2 = layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), activation=tf.keras.activations.relu,padding='valid')(conv1)
            conv3 = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation=tf.keras.activations.relu,padding='valid')(conv2)
            flat = layers.Flatten()(conv3)
            
            #add to layer lists
            input_layer_list.append(image_input)
            dense_layer_list.append(flat)
            
        merged_dense = tf.keras.layers.concatenate(dense_layer_list)
        
        advantage_fc = layers.Dense(512, activation='relu')(merged_dense)
        advantage = layers.Dense(num_actions)(advantage_fc)
        advantage = layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),
                           output_shape=(num_actions,))(advantage)

        value_fc = layers.Dense(512, activation='relu')(merged_dense)
        value =  layers.Dense(1)(value_fc)
        value = layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),
                       output_shape=(num_actions,))(value)

        q_values = layers.Add()([value, advantage])

        model = tf.keras.Model(inputs = input_layer_list, outputs = q_values)
        return model

    def DUELING_LSTM(self,image_shape,num_actions, trace_length):
        #from dueling example
        image_input = tf.keras.Input(shape=(trace_length,) + image_shape)
        conv1 = layers.TimeDistributed(layers.Conv2D(32, kernel_size=(8,8), strides=(4,4), activation=tf.keras.activations.relu,padding='valid'))(image_input)
        conv2 = layers.TimeDistributed(layers.Conv2D(64, kernel_size=(4,4), strides=(2,2), activation=tf.keras.activations.relu,padding='valid'))(conv1)
        conv3 = layers.TimeDistributed(layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), activation=tf.keras.activations.relu,padding='valid'))(conv2)
        flatten = layers.TimeDistributed(layers.Flatten())(conv3)
        
        lstm = layers.LSTM(512)(flatten)

        advantage_fc = layers.Dense(512, activation='relu')(lstm)
        advantage = layers.Dense(num_actions)(advantage_fc)
        advantage = layers.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True),
                           output_shape=(num_actions,))(advantage)

        value_fc = layers.Dense(512, activation='relu')(lstm)
        value =  layers.Dense(1)(value_fc)
        value = layers.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1),
                       output_shape=(num_actions,))(value)
       
        q_values = layers.Add()([value, advantage])
        
        model = tf.keras.Model(inputs=image_input, outputs=q_values)
        return model