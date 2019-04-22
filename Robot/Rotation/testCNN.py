import environment
from environment import Location,Reward,Action
import tensorflow as tf
import numpy as np

test_iterations = 20
max_moves = 200
wins  = 0
losses = 0
delay = 500
env= environment.Environment(width=640,height=480,random_location=False,mineral_scale=.5,camera_height=3.5,camera_tilt=0,start_shift=15,start_pos=23.5,actions=[Action.FORWARDS,Action.CW,Action.CCW],reward=Reward.RELATIVE_PROPORTIONAL,decorations=True,resize_scale=16,x_collision_scale=3,y_collision_scale=3,silver=(.8,.8,.8),random_colors=True,random_lighting=True)
multi = False
num_actions = env.action_space()
image_shape = np.shape(env.screenshot())
image_len = len(image_shape)

#no longer used, but older models may have loss set as q_loss
def q_loss(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred))

with tf.device("/GPU:0"):
    #model = tf.keras.models.load_model('./models/ff1.h5',custom_objects={ 'q_loss': q_loss})
    model = tf.keras.models.load_model('./models/cnnrandomRCPT.h5')

    
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
  
for i in range(test_iterations):
    if multi:
            state = env.full_reset()
    else:
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