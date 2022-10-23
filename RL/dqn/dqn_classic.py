import random
import math
import gym
import numpy as np
import time
from collections import deque

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense

tensorflow.enable_eager_execution()


class Buffer:
    def __init__(self, n):
        self.buf = deque(maxlen=n)

    def push(self, x):
        self.buf.append(x)

    def pop(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))

    def get_size(self):
        return len(self.buf)


def select_action_eps_greedy(env, s, eps, nn):
    acts = nn.predict(s)[0]
    max_q = max(acts)
    if np.random.random() <= eps:
        return max_q, env.action_space.sample()
    else:
        best_as = []
        for i in range(env.action_space.n):
            if acts[i] == max_q:
                best_as.append(i)
        return max_q, random.choice(best_as)


class NN:
    def __init__(self, input_size, output_size, hidden_layer_sizes):
        self.model = Sequential()
        self.model.add(Dense(hidden_layer_sizes[0], input_dim=input_size, activation='tanh'))
        for s in hidden_layer_sizes[1:]:
            self.model.add(Dense(s, activation='tanh'))
        self.model.add(Dense(output_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.01, decay=0.01))

    def copy_from(self, other):
        self.model.set_weights(other.model.get_weights())

    def get_max(self, s):
        return np.max(self.predict(s)[0])

    def predict(self, s):
        s = np.atleast_2d(s)
        return self.model.predict(s)

    def sgd_step(self, x_batch, y_batch):
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)


def get_eps(t, eps, eps_min, eps_decay):
    return max(eps_min, min(eps, 1.0 - math.log10((t + 1) * eps_decay)))


def dqn(env_name,
        mem_capacity=50000,
        min_buffer_size=100,
        episodes_num=1000,
        steps_num=5000,
        batch_size=32,
        discount=1.,
        episodes_till_update=10,
        last_100_av_reward_to_win=190.0):
    env = gym.make(env_name)

    try:
        buffer = Buffer(mem_capacity)

        input_size = len(env.observation_space.sample())
        actions_num = env.action_space.n
        layer_sizes = [20, 40]
        dqn = NN(input_size, actions_num, layer_sizes)
        target_dqn = NN(input_size, actions_num, layer_sizes)
        target_dqn.copy_from(dqn)

        eps = 1.
        eps_decay = 0.995
        eps_min = 0.01

        rewards = []
        av_max_qs = []

        for i_episode in range(episodes_num):
            state = env.reset()
            total_reward = 0.
            total_max_q = 0.
            time_steps = 0
            done = False
            cur_eps = get_eps(i_episode, eps, eps_min, eps_decay)
            while not done:
                #                 env.render()
                max_q, action = select_action_eps_greedy(env, state, cur_eps, dqn)
                next_state, reward, done, info = env.step(action)
                buffer.push((state, action, reward, next_state, done))
                state = next_state

                total_reward += reward
                total_max_q += max_q
                time_steps += 1

            if buffer.get_size() >= min_buffer_size:
                batch = buffer.pop(batch_size)
                target_qs = []
                states = []
                for (s, a, r, next_s, is_terminal) in batch:
                    states.append(s)
                    target_predict = dqn.predict(s)[0]
                    target_predict[a] = r if is_terminal else r + discount * target_dqn.get_max(next_s)
                    target_qs.append(target_predict)
                dqn.sgd_step(states, target_qs)

            if eps > eps_min:
                eps *= eps_decay

            if i_episode % episodes_till_update == 0:
                target_dqn.copy_from(dqn)

            rewards.append(total_reward)
            av_max_qs.append(total_max_q / time_steps)

            last_100_av_reward = np.array(rewards[max(0, i_episode - 100):(i_episode + 1)]).mean()
            if i_episode % 100 == 0:
                print("episode:", i_episode, "total reward:", total_reward, "eps:", round(eps, 3),
                      "avg reward (last 100):", round(last_100_av_reward, 2))
            if (i_episode + 1) >= 100 and last_100_av_reward >= last_100_av_reward_to_win:
                print("Solved after", i_episode, "episodes with total reward:", total_reward,
                      "and avg reward (last 100):", round(last_100_av_reward, 2))
                break

        print("avg reward for last 100 episodes:", np.array(rewards[-100:]).mean())
    except:
        raise
    finally:
        env.close()
    return dqn, rewards, av_max_qs


def test(env_name, nn):
    env = gym.make(env_name)
    eps = 0.01
    state = env.reset()
    while not done:
        action = select_action_eps_greedy(env, state, eps, nn)
        state, _, done, _ = env.step(action)
        env.render()
    env.close()


def save(path, agents_rewards, av_max_qs, nn):
    for i in range(9):
        with open(f'{path}/rewards' + str(i), 'w') as f:
            np.savetxt(f, agents_rewards[i])
        with open(f'{path}/qs' + str(i), 'w') as f:
            np.savetxt(f, av_max_qs[i])
    nn.model.save(f'{path}/model')


def load(path):
    rewards = []
    qs = []
    for i in range(9):
        with open(f'{path}/rewards' + str(i), 'r') as f:
            rewards.append(np.loadtxt(f))
        with open(f'{path}/qs' + str(i), 'r') as f:
            qs.append(np.loadtxt(f))
    return rewards, qs

