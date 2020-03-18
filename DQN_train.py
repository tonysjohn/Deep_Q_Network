import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam,SGD
from collections import deque
import time
import numpy as np
import random
from tqdm import tqdm

#DQN Settings
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 200
MINIBATCH_SIZE = 32
DISCOUNT =0.99
UPDATE_TARGET_EVERY = 5
MEMORY_FRACTION = 0.20
NUM_EPOCH=2
LEARNING_RATE_DECAY = 0.9

# Min Reward settings for Saving
MIN_REWARD = 650

# Environment settings
EPISODES = 1000
ENV ='MountainCar-v0'

# Exploration settings
EPSILON = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.01

#  Stats settings
AGGREGATE_STATS_EVERY = 1  # episodes
SHOW_PREVIEW = False

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #tf.compat.v1.disable_eager_execution()
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        pass
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for key,value in stats.items():
                tf.summary.scalar(key,value,self.step)

class DQNAgent:
    def __init__(self, environment):

        #Env Variables
        self.env = gym.make(environment)
        self.observation_size =  self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        # main model
        self.model = self.createModel()

        # Target model
        self.target_model = self.createModel()
        self.target_model.set_weights(self.model.get_weights())

        #Tensorboard
        self.tensorboard = ModifiedTensorBoard(log_dir = f"log/{ENV}-{int(time.time())}")
        
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0

    def createModel(self):
        model = Sequential()
        model.add(Dense(24, input_shape = [self.observation_size]))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.2))
        model.add(Dense(48))
        model.add(Activation('relu'))
        #model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr=0.001), metrics = ['accuracy'])
        return model
    

    # Memory buffer
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    # Predict action values    
    def get_qs(self, state):
        return np.array(self.model.predict_on_batch(state))[0]

    
    def minibatch(self):
        
        #Random minibatch sampling
        minibatch = random.sample(self.replay_memory,MINIBATCH_SIZE)

        #Current MOdel prediction
        current_states = np.array([transition[0] for  transition in minibatch]).reshape(MINIBATCH_SIZE, self.observation_size)
        current_qs_list = np.array(self.model.predict_on_batch(current_states))

        #Target model prediction
        new_current_states = np.array([transition[3] for transition in minibatch]).reshape(MINIBATCH_SIZE, self.observation_size)
        future_qs_list = np.array(self.target_model.predict_on_batch(new_current_states))

        X=[]
        y=[]

        for index, (current_state, action, reward, new_current_states, done) in enumerate(minibatch):

            # update Q value of State-Action pair in current model
            if not done:
                max_feature_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT*max_feature_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        return np.array(X).reshape(MINIBATCH_SIZE, self.observation_size), np.array(y)
        

    # Train DQN
    def train(self):

        # Do nothing if low buffer
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
            
        X, y = self.minibatch()

        for i in range(NUM_EPOCH):
            loss = self.model.train_on_batch(X, y) #train_on_batch is faster than fit
        self.target_update_counter +=1

        #Sync Target model
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter=0

        return loss
    
    def best_action(self,current_state,epsilon):
            # Epsilon greedy
        if np.random.random() > epsilon:
            # Get action from Q table
            actions=self.get_qs(current_state)
            return np.argmax(actions)
        else:
            # Get random action
            return np.random.randint(0, self.action_size)

# def custom_reward(reward,state):
#     reward += abs(state[0]-start)*10
#     # #print(reward)
#     if state[0]== 0.5:
#         reward+=200
#     return reward

def main():

    agent = DQNAgent(ENV)
    ep_rewards = []
    epsilon=EPSILON

    for episode in tqdm(range(1, EPISODES+1), ascii=True, unit = 'episodes'):
        agent.tensorboard.step = episode
        episode_reward=0
        current_state = agent.env.reset()
        done = False
        current_state= current_state.reshape(1, agent.observation_size)

        while not done:
            #print(current_state)
            action = agent.best_action(current_state,epsilon)

            #Getting new states
            new_state, reward, done, _ = agent.env.step(action)

            #reward = custom_reward(reward,new_state)
            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:  #Render Environment
                agent.env.render()

            new_state = new_state.reshape(1, agent.observation_size)

            #Add to buffer
            agent.update_replay_memory((current_state, action, reward, new_state, done))

            #Train DQN
            loss = agent.train()
            current_state = new_state

        ep_rewards.append(episode_reward)

        
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            median_reward = np.median(ep_rewards[-AGGREGATE_STATS_EVERY:]) #sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(loss=loss[0], accuracy=loss[1], reward_median=median_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
            # print('median reward={}, min reward={}, max reward={}, epsilon={}'.format(median_reward,min_reward,max_reward,epsilon))

                # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models\{ENV}__{max_reward:_>7.2f}max_{median_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
                
                #reduce learning if required
                tf.keras.backend.set_value(agent.model.optimizer.learning_rate, tf.keras.backend.get_value(agent.model.optimizer.learning_rate)*LEARNING_RATE_DECAY)
        
        #Epsilon Decay
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    #print(ep_rewards)
    agent.env.close()

        
if __name__ == "__main__":
    main()