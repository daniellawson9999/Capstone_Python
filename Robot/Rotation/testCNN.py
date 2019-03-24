import environment
from environment import Location,Reward,Action
import tensorflow as tf
import numpy as np

test_iterations = 100
max_moves = 200
wins  = 0
losses = 0
delay = 500
env = environment.Environment(random_minerals = True,mineral_scale=1,random_location=False,reward=Reward.RELATIVE_PROPORTIONAL,start_shift=-3,camera_height=5,actions=[Action.FORWARDS,Action.CW,Action.CCW])

num_actions = env.action_space()
image_shape = np.shape(env.screenshot())
image_len = len(image_shape)
def q_loss(y_true, y_pred):
    #y_true = Q(s), y_pred = y
   # y_true = tf.keras.backend.placeholder(ndim = 1, dtype = 'float32', name = 'y_true')
    #y_pred = tf.keras.backend.placeholder(ndim = 2, dtype = 'float32', name = 'y_pred')
    #q = tf.keras.backend.max(y_pred)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))
    #return tf.math.reduce_sum(tf.math.squared_difference(y_true,y_pred)) / 2
#np.argmax(model.predict(np.expand_dims(state,0))) 

with tf.device("/GPU:0"):
    #model = tf.keras.models.load_model('./models/ff1.h5',custom_objects={ 'q_loss': q_loss})
    model = tf.keras.models.load_model('./models/cnnrandomS1.h5')

    
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
#new_model.summary()
env.random_location = False  
for i in range(test_iterations):
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