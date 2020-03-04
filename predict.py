import gym
import tensorflow as tf
import numpy as np
from enivornment import DQNAgent


def main():
    agent = DQNAgent()
    env = gym.make('MountainCar-v0')

    N = env.observation_space.shape[0]

    load_model = tf.keras.models.load_model(r'C:\Files and Folders\project\RL\models\MountainCar-v0__1159.30max_1159.30avg_1159.30min__1583294343.model')
    agent.model.set_weights(load_model.get_weights())
    
    for i in range(20):

        allreward=[]
        episode_reward=0
        current_state = env.reset()
        done = False

        while not done:
            env.render()

            current_state=current_state.reshape(1,N)
            actions=agent.get_qs(current_state)

            action = np.argmax(actions)
            print(action)

            new_state, reward, done, _ = env.step(action)
            current_state=new_state
            episode_reward+=reward
        allreward.append(episode_reward)

    print(np.mean(np.array(allreward)))
    env.close()

if __name__ == "__main__":
    main()
