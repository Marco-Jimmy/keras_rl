import keras
import gym
import random
import numpy as np
from collections import deque

MAX_MEMORY = 2000
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.95

EPISODES = 1000
TIME = 200


class Agent_DQN:
    def __init__(self, state_size, action_size):
        self.learning_rate = LEARNING_RATE
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = self._init_model()
        self.gamma = DISCOUNT_RATE
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _init_model(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(10, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(optimizer=keras.optimizers.Adam(lr=self.learning_rate), loss='mse')
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def replay(self, batch_size):
        random_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in random_batch:

            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, verbose=0, epochs=1)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent_DQN(state_size, action_size)
    batch_size = 32
    done = False

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(TIME):
            if e % 10 == 0:
                env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, time: {}, e: {}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
