import environment
from environment import Action, Reward, Location
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
from time import time

epsilon = 1
gamma = .95
alpha = .0001

iterations = 2000
decay_rate = 1/iterations
test_iterations = 10
max_moves =  80
win_reward = 100 
loss_reward = -win_reward


max_memory_size = iterations * max_moves
batch_size = 16


wins = 0
losses = 0
reached_max = 0
reward_list = []

training_win = 0
training_loss = 0

load_model = True
load_name = 'cnnrandomturnp800.h5' 

#env = environment.Environment(random_minerals=True,random_location=False,mineral_location=Location.RIGHT,reward=Reward.RELATIVE_PROPORTIONAL,actions=[Action.FORWARDS,Action.LEFT,Action.RIGHT])
env= environment.Environment(width=640,height=480,random_location=False,mineral_scale=.5,
                             camera_height=3.5,camera_tilt=0,start_shift=15,start_pos=23.5,
                             actions=[Action.FORWARDS,Action.CW,Action.CCW],
                             reward=Reward.RELATIVE_PROPORTIONAL,decorations=True,
                             resize_scale=16,x_collision_scale=3,y_collision_scale=3,
                             silver=(.8,.8,.8),random_colors=True,random_lighting=True)
env.loss_reward = loss_reward
env.win_reward = win_reward

image_shape = np.shape(env.screenshot())
image_len = len(image_shape)
if image_len == 2:
    image_shape = image_shape + (1,)
    
num_actions = env.action_space()

evaluate_training = True
save_model = True
model_name = "test.h5"

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
    

    
memory = ReplayMemory(max_memory_size)

#initialize memory to batch size
    
while len(memory.memory) < batch_size:
    state = env.reset()
    done = False
    for i in range(max_moves):
        action = env.sample()
        next_state, reward, done,_ = env.step(action)
        transition = [state, action, env.legal_actions(),reward,next_state,int(done)]
        memory.store(transition)
        state = next_state
        if len(memory.memory) >= batch_size: break


#q network

#legacy
def q_loss(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))
#define the nn structure
with tf.device("/GPU:0"):
    #fc nn
    #inputs = tf.keras.Input(shape= [image_shape[0] + env.action_space()] )
    #flat = tf.keras.layers.Flatten()(inputs) 
    #hidden1 = tf.keras.layers.Dense(int(np.round(np.product(image_shape[0])/30)), activation= tf.keras.activations.relu)(inputs)
    #hidden2 = tf.keras.layers.Dense(int(np.round(np.product(image_shape[0])/30)), activation= tf.keras.activations.relu)(hidden1)
    #outputs= tf.keras.layers.Dense(1, activation= tf.keras.activations.linear)(hidden2)
    #model =  tf.keras.Model(inputs=inputs, outputs=outputs)
    if load_model:
        model = tf.keras.models.load_model('./models/cnnrandomP1.h5')
    #state cnn combined with fc action to create a sa convnet
    else:
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
        
        model = tf.keras.Model(inputs=[image_input,action_input], outputs = output)
        model.compile(loss = tf.keras.losses.mean_squared_error ,optimizer = tf.keras.optimizers.Adam(lr = alpha))

    #conv net
    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation=tf.keras.activations.relu, input_shape=image_shape,strides=2))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=2, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.Dense(env.action_space(), activation=tf.keras.activations.linear))
    
    #1d option, bad idea
    #model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Conv1D(32, kernel_size=1, activation=tf.keras.activations.relu, input_shape=(1,image_shape[0] + env.action_space()),strides=1))
    #model.add(tf.keras.layers.Conv1D(32, kernel_size=1, strides=1, activation='relu'))
    #model.add(tf.keras.layers.MaxPooling1D(pool_size13))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    #model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

#define loss function 
#def q_loss(state):
#    #unused but necessary? test
#    def loss(y_true, y_pred):
#        action,reward,next_state,terminal = memory.get_full_transition(state)
#        if terminal:
#            y = reward
#        else:
#            y = reward + gamma * max(model.predict(next_state)) #reward + gamma * max(Q(s'))
#        return tf.keras.squa  (y - max(model.predict(state)))**2 # (y- max(Q(s)))^2
#    return loss;  


    
def predict(state,legal_actions = env.legal_actions()):
    actions = [0] * num_actions
    for i in range(num_actions):
        action = [0] * num_actions
        action[i] = 1
        if image_len == 2:
            actions[i] = model.predict([np.expand_dims(np.expand_dims(state,2),0),np.expand_dims(action,0)])[0]
        else:
            actions[i] = model.predict([np.expand_dims(state,0),np.expand_dims(action,0)])[0]
    max_index = 0
    for i in range(len(actions)):
        if legal_actions[max_index] == 0 and legal_actions[i] == 1:
            max_index = i
        if actions[max_index][0] < actions[i][0] and legal_actions[i] == 1:
            max_index = i
    return max_index, actions[max_index][0]

#tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/tb',batch_size = batch_size)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()),batch_size = batch_size,   write_grads=True,
    write_images=True)
tensorboard.set_model(model)

#begin training
count = 0
for i in range(iterations):
    print("starting iteration ",i)
    total_reward = 0
    state = env.reset()
    #temp
    while env.state() != environment.State.STANDARD:
        state = env.reset()
    for m in range(max_moves):
        #select an action
        if np.random.random() > epsilon:
            action, value = predict(state,env.legal_actions())

        else:
            action = env.sample()
        #transition
        next_state, reward, done,game_state = env.step(action)
        total_reward += reward
        #store transition
        transition = [state,action,env.legal_actions(),reward,next_state,int(done)]
        memory.store(transition)
        #sample batch of transitions from memory
        batch = memory.sample(batch_size)
        #states = np.asarray(tuple(batch[:,0]))
        states = list(np.zeros(batch_size))
        actions = list(np.zeros(batch_size))
        #obtain the targets, y
        targets = list(np.zeros(batch_size))
        for j in range(batch_size):
            a = batch[j,1]
            onehot = [0] * num_actions
            onehot[a] = 1
            actions[j] = onehot
            s = batch[j,0]
            states[j] = s
            n_s = batch[j,4]
            l = batch[j,2]
            r = batch[j,3]
            t = batch[j,5]
            if t:
                targets[j] = r
            else:
                index, value = predict(n_s,l)
                targets[j] = r + gamma * value
        #perform gradient descent w/ batch
        states = np.asarray(states)
        if image_len == 2:
            states = np.expand_dims(states,3)
        actions = np.asarray(actions)
        #model.fit()
        loss = model.train_on_batch(x = [states,actions], y = targets)
        logs = {}
        logs['loss'] = loss
        tensorboard.on_epoch_end(count, logs)
        count += 1
        #update state
        state = next_state
        if done or m == max_moves - 1:
            if game_state == environment.State.WIN:
                training_win += 1
            if game_state == environment.State.LOSS:
                training_loss += 1
            print("total reward {} last iteration {} moves, total wins {}, total losses {}".format(total_reward,m+1,training_win,training_loss))
            #print("Episode finished after {} timesteps".format(t+1))
            print("epsilon ",epsilon)
            #decrease epsilon linearlly until .1
            epsilon = max(epsilon - decay_rate, .1)
            reward_list.append(total_reward/(m+1))
            #print(epsilon)
            break
tensorboard.on_train_end(None)
plt.plot(reward_list)
plt.ylabel('average_reward')
plt.xlabel('episode')
plt.show()
#evaluate
if evaluate_training:
    env.random_location = False
    for i in range(test_iterations):
        state = env.reset()
        #temp
        while env.state() != environment.State.STANDARD:
            state = env.reset()
        for t in range(max_moves):
            action,value = predict(state,env.legal_actions())
            #action = env.action_space.sample()
            state, reward, done, game_state = env.step(action)
            if done:
                if game_state == environment.State.WIN:
                    wins += 1
                elif game_state == environment.State.LOSS:
                    losses += 1
                break
    print("{} wins, {} losses, {} reached max".format(wins/test_iterations, losses/test_iterations,(test_iterations-wins-losses)/test_iterations))

if save_model:
    model.save('./models/' + model_name)