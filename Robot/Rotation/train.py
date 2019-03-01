import environment
import tensorflow as tf
import numpy as np
import collections
import matplotlib.pyplot as plt
from time import time
epsilon = 1
gamma = .95
alpha = .001

iterations = 100
decay_rate = 1/iterations
test_iterations = 25
max_moves =  85
win_reward = max_moves * 3
loss_reward = -win_reward


max_memory_size = iterations * max_moves
batch_size = 20


wins = 0
losses = 0
reached_max = 0
reward_list = []

training_win = 0
training_loss = 0 

env = environment.Environment()
env.loss_reward = loss_reward
env.win_reward = win_reward

image_shape = np.shape(env.screenshot())

evaluate_training = True
save_model = True

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
    #inputs = tf.keras.Input(shape=image_shape)
    #flat = tf.keras.layers.Flatten()(inputs) 
    #hidden1 = tf.keras.layers.Dense(int(np.round(np.product(image_shape)/30)), activation= tf.keras.activations.relu)(flat)
    #outputs= tf.keras.layers.Dense(env.action_space(), activation= tf.keras.activations.linear)(hidden1)
    #model =  tf.keras.Model(inputs=inputs, outputs=outputs)
    #model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(image_shape)),
    #tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    #tf.keras.layers.Dense(env.action_space(), activation=tf.keras.activations.linear)
#])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation=tf.keras.activations.relu, input_shape=image_shape,strides=2))
    #model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation=tf.keras.activations.relu, input_shape=image_shape,strides=2))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(128, activation=tf.keras.activations.relu))
    model.add(tf.keras.layers.Dense(env.action_space(), activation=tf.keras.activations.linear))


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
    #y_true = Q(s), y_pred = y
   # y_true = tf.keras.backend.placeholder(ndim = 1, dtype = 'float32', name = 'y_true')
    #y_pred = tf.keras.backend.placeholder(ndim = 2, dtype = 'float32', name = 'y_pred')
    #q = tf.keras.backend.max(y_pred)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))
    #return tf.math.reduce_sum(tf.math.squared_difference(y_true,y_pred)) / 2
#np.argmax(model.predict(np.expand_dims(state,0))) 
def predict(state,legal_actions = env.legal_actions()):
    actions = model.predict(np.expand_dims(state,0))[0]
    max_index = 0
    for i in range(np.size(actions)):
        if legal_actions[max_index] == 0 and legal_actions[i] == 1:
            max_index = i
        if actions[max_index] < actions[i] and legal_actions[i] == 1:
            max_index = i
    return max_index

def named_logs(model,logs):
    result = {}
    if(len(model.metrics_names) == 1):
        result[model.metrics_names[0]] = logs
    else:    
        for l in zip(model.metrics_names,logs):
            result[l[0]] = l[1]
    return result   
     
model.compile(loss = q_loss ,optimizer = tf.keras.optimizers.Adam(lr=alpha))
#tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/tb',batch_size = batch_size)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(time()),batch_size = batch_size)
tensorboard.set_model(model)

#begin training
for i in range(iterations):
    print("starting iteration ",i)
    total_reward = 0
    state = env.reset()
    a = []
    for m in range(max_moves):
        #select an action
        if np.random.random() > epsilon:
            action = predict(state,env.legal_actions())

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
        states = np.asarray(tuple(batch[:,0]))
        #obtain the targets, y
        targets = np.zeros(batch_size)
        for j in range(batch_size):
            s = batch[j,0]
            a = batch[j,1]
            l = batch[j,2]
            r = batch[j,3]
            n_s = batch[j,4]
            t = batch[j,5]
            if t:
                targets[j] = r
            else:
                targets[j] = r + gamma * predict(n_s,l)
        #perform gradient descent w/ batch
        logs = model.train_on_batch(x = states, y = targets)
        tensorboard.on_epoch_end(i, named_logs(model,logs))
        
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
            reward_list.append(total_reward)
            #print(epsilon)
            break
tensorboard.on_train_end(None)
plt.plot(reward_list)
plt.ylabel('total_reward')
plt.xlabel('episode')
plt.show()
#evaluate
if evaluate_training:
    for i in range(test_iterations):
        state = env.reset()
        for t in range(max_moves):
            action = predict(state,env.legal_actions())
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
    model.save('./models/cnn2.h5')