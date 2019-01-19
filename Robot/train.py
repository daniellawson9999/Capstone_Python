import environment
import tensorflow as tf
import numpy as np
import collections

epsilon = 1
gamma = .6
alpha = .7

iterations = 1000
decay_rate = 1/iterations
test_iterations = 1000

max_memory_size = iterations * 2
batch_size = 10

wins = 0
draws = 0

env = environment.Environment()

def convert_to_int(state):
    return (state[0],state[1],int(state[2]))

 #based off deque example
class ReplayMemory():
    def __init__(self, size):
        self.memory = collections.deque(maxlen = size)
        
    def store(self, transition):
        #transition should be an array of s,a,r,s',t
        self.memory.append(transition)
        
    def sample(self,n):
        memory_size = len(self.memory)
        return np.asarray([self.memory[i] for i in 
                (np.random.choice(np.arange(memory_size),size=n, replace=False))])
    

    
memory = ReplayMemory(iterations*2)

#initialize memory to batch size
    
while len(memory.memory) < batch_size:
    state = convert_to_int(env.reset())
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        next_state = convert_to_int(next_state)
        transition = [state, action,reward,next_state,int(done)]
        memory.store(transition)
        state = next_state



#q network


#define the nn structure
inputs = tf.keras.Input(shape=(3,))
hidden = tf.keras.layers.Dense(3, activation= tf.keras.activations.relu)(inputs)
outputs= tf.keras.layers.Dense(2, activation= tf.keras.activations.linear)(hidden)
model =  tf.keras.Model(inputs=inputs, outputs=outputs)


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

model.compile(loss = q_loss,optimizer = 'SGD')

#begin training
for i in range(iterations):
    state = convert_to_int(env.reset())
    
    for t in range(10):
        #select an action
        if np.random.random() > epsilon:
            action = np.argmax(model.predict(np.expand_dims(state,0)))

        else:
            action = env.action_space.sample()
        #transition
        next_state, reward, done, info = env.step(action)
        next_state = convert_to_int(next_state)
        #store transition
        transition = [state,action,reward,next_state,int(done)]
        memory.store(transition)
        #sample batch of transitions from memory
        batch = memory.sample(batch_size)
        states = np.asarray(tuple(batch[:,0]))
        #obtain the targets, y
        targets = np.zeros(batch_size)
        for i in range(batch_size):
            s = batch[i,0]
            a = batch[i,1]
            r = batch[i,2]
            n_s = batch[i,3]
            t = batch[i,4]
            if t:
                targets[i] = r
            else:
                targets[i] = r + gamma * np.max(model.predict(np.expand_dims(n_s,0)))
        #perform gradient descent w/ batch
        model.train_on_batch(x = states, y = targets)
        #update state
        state = next_state
        if done:
            #print("Episode finished after {} timesteps".format(t+1))
            epsilon -= decay_rate
            break
#evaluate
for i in range(test_iterations):
    state = convert_to_int(env.reset())
    for t in range(10):
        action = np.argmax(model.predict(np.expand_dims(state,0)))
        #action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        state = convert_to_int(next_state)
        if done:
            if reward == 1:
                wins += 1
            elif reward == 0:
                draws += 1
            break
losses = test_iterations - wins - draws
print("{} win, {} draw, {} loss".format(wins/test_iterations, draws/test_iterations,losses/test_iterations))
