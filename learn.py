
#environment object
env = Environment()

#parameters
actions = env.po
number_actions = env.action_space.n
state_size = env.observation_space.n
scale =
processed_size = state_size / scale


gamma =
alpha =

episodes =
steps =
#handle initiial batch size situation
batch_size =

epsilon =
decay_rate = epsilon / episodes

max_memory =


#Memory object to handle adding memory and sampling
#find best python collection for this
memory = Memory(max_memory,number_actions,processed_size)

nn = NN(number_actions, number_states, alpha)

for e in range(episodes):
    state = env.reset()
    for i in range(steps):
        env.render()
        #maybe make > into a function

        if random() > epsilon:
            action = env.action_space.sample()
        else:
            actions_values = predict(state)
            action = np.argmax(action_values)

        next_state, reward, done = env.step(action)
        next_state = process(next_state,processed_size)

        memory.append(state,next_state, action, reward)
        batch = memory.getBatch(batch_size)
        predicted_values = batchValue(batch)
        '''
        y =
            reward if terminal state
            reward + gamma * max(next_state) else
        '''
        nn.gradientDescent(batch.actions, )
