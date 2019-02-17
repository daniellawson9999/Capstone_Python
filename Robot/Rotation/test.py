import environment
import tensorflow as tf
import numpy as np

test_iterations = 50
max_moves = 300
wins  = 0
losses = 0

env = environment.Environment()

def q_loss(y_true, y_pred):
    #y_true = Q(s), y_pred = y
   # y_true = tf.keras.backend.placeholder(ndim = 1, dtype = 'float32', name = 'y_true')
    #y_pred = tf.keras.backend.placeholder(ndim = 2, dtype = 'float32', name = 'y_pred')
    #q = tf.keras.backend.max(y_pred)
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))
    #return tf.math.reduce_sum(tf.math.squared_difference(y_true,y_pred)) / 2
#np.argmax(model.predict(np.expand_dims(state,0))) 

with tf.device("/GPU:0"):
    model = tf.keras.models.load_model('robot.h5',custom_objects={ 'q_loss': q_loss})
    
def predict(state,legal_actions = env.legal_actions()):
    actions = model.predict(np.expand_dims(state,0))[0]
    max_index = 0
    for i in range(np.size(actions)):
        if legal_actions[max_index] == 0 and legal_actions[i] == 1:
            max_index = i
        if actions[max_index] < actions[i] and legal_actions[i] == 1:
            max_index = i
    return max_index   
#new_model.summary()
    
for i in range(test_iterations):
    state = env.reset(random = False)
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