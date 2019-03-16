import environment
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
from time import time
epsilon = 1
gamma = .95
alpha = .001

iterations = 10
decay_rate = 1/iterations
test_iterations = 10
max_moves =  100
win_reward = max_moves * 3
loss_reward = -win_reward


max_memory_size = iterations * max_moves
batch_size = 10


wins = 0
losses = 0
reached_max = 0
reward_list = []

training_win = 0
training_loss = 0 

env = environment.Environment(random_minerals=False,random_location=False,mineral_location=environment.Location.RIGHT,reward=environment.Reward.RELATIVE)
env.loss_reward = loss_reward
env.win_reward = win_reward

image_shape = np.shape(env.screenshot())

evaluate_training = True
save_model = True
model_name = "fixed-right-fc"

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


#define the nn structure
with tf.device("/GPU:0"):
    #fc nn
    #inputs = tf.keras.Input(shape= [image_shape[0] + env.action_space()] )
    #flat = tf.keras.layers.Flatten()(inputs) 
    #hidden1 = tf.keras.layers.Dense(int(np.round(np.product(image_shape[0])/30)), activation= tf.keras.activations.relu)(inputs)
    #hidden2 = tf.keras.layers.Dense(int(np.round(np.product(image_shape[0])/30)), activation= tf.keras.activations.relu)(hidden1)
    #outputs= tf.keras.layers.Dense(1, activation= tf.keras.activations.linear)(hidden2)
    #model =  tf.keras.Model(inputs=inputs, outputs=outputs)
    
    #state cnn combined with fc action to create a sa convnet
    image_input = tf.keras.Input(shape=image_shape)
    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation=tf.keras.activations.relu, input_shape=image_shape,strides=2)(image_input)
    drop1 = tf.keras.layers.Dropout(0.5)(conv1)
    conv2 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=2, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.5)(conv2)
    flat = tf.keras.layers.Flatten()(drop2)
    conv_dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(flat)
    
    action_input = tf.keras.Input(shape=(env.action_space(),))
    action_dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)(action_input)
    
    merged = tf.keras.layers.concatenate([conv_dense, action_dense])
    output = tf.keras.layers.Dense(1,activation=tf.keras.activations.linear)(merged)
    
    model = tf.keras.Model(inputs=[image_input,action_input], outputs = output)
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

def q_loss(y_true, y_pred):
    #y_true and y_pred are both full batch size (20x6 and)
    #y_true = Q(s), y_pred = y
   # y_true = tf.keras.backend.placeholder(ndim = 1, dtype = 'float32', name = 'y_true')
    #y_pred = tf.keras.backend.placeholder(ndim = 2, dtype = 'float32', name = 'y_pred')
    #q = tf.keras.backend.max(y_pred)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))
    #return tf.math.reduce_sum(tf.math.squared_difference(y_true,y_pred)) / 2
#np.argmax(model.predict(np.expand_dims(state,0))) 

    
def predict(state,legal_actions = env.legal_actions()):
    copy_state = list(np.copy(state))
    #actions = model.predict(np.expand_dims(state,0))[0]
    actions = [0,0,0,0,0,0]
    for i in range(6):
        copy_state.append(0)
    for i in range(6):
        if i != 0:
            copy_state[len(copy_state)-6-1+i] = 0
        copy_state[len(copy_state)-6+i] = 1
        actions[i] = model.predict(np.asarray(np.expand_dims(copy_state,0)))[0]
    max_index = 0
    for i in range(len(actions)):
        if legal_actions[max_index] == 0 and legal_actions[i] == 1:
            max_index = i
        if actions[max_index][0] < actions[i][0] and legal_actions[i] == 1:
            max_index = i
    return max_index, actions[max_index][0]
     
model.compile(loss = tf.keras.losses.mean_squared_error ,optimizer = tf.keras.optimizers.Adam(lr = alpha))
#tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/tb',batch_size = batch_size)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()),batch_size = batch_size)
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
        #obtain the targets, y
        targets = list(np.zeros(batch_size))
        for j in range(batch_size):
            a = batch[j,1]
            s = list(batch[j,0])
            for i in range(6):
                if i == a:
                    v = 1
                else: 
                    v = 0
                s.append(v)
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
        loss = model.train_on_batch(x = states, y = targets)
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
            epsilon -= decay_rate
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
    for i in range(test_iterations):
        state = env.reset()
        #temp
        while env.state() != environment.State.STANDARD:
            state = env.reset()
        for t in range(max_moves):
            action,value = predict(state,env.legal_actions())
            #action = env.action_space.sample()
            next_state, reward, done, game_state = env.step(action)
            if done:
                if game_state == environment.State.WIN:
                    wins += 1
                elif game_state == environment.State.LOSS:
                    losses += 1
                break
    print("{} wins, {} losses, {} reached max".format(wins/test_iterations, losses/test_iterations,(test_iterations-wins-losses)/test_iterations))

if save_model:
    model.save('./models/' + model_name)