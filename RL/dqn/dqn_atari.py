import random
import math
import gym
import numpy as np
import time
import cv2
import gc
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.enable_eager_execution()


class ImageMemory:
    def __init__(self, max_size, frame_height, frame_width, batch_size,
                 agent_history_length=4):
        self.size = max_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.agent_history_length = agent_history_length
        self.batch_size = batch_size
        self.count = 0
        self.current = 0

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.short)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.dones = np.empty(self.size, dtype=np.bool)

        # Pre-allocate memory for the states and new_states in a minibatch
        self.states = np.empty((self.batch_size, self.agent_history_length,
                                self.frame_height, self.frame_width), dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.agent_history_length,
                                    self.frame_height, self.frame_width), dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.dones[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def _get_state(self, index):
        return self.frames[index - self.agent_history_length + 1:index + 1, ...]

    def setup_valid_indices(self):
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.agent_history_length, self.count - 1)
                if index >= self.current and index - self.agent_history_length <= self.current:
                    continue
                if self.dones[index - self.agent_history_length:index].any():
                    continue
                break
            self.indices[i] = index

    def get_mini_batch(self):
        self.setup_valid_indices()

        for i in range(len(self.indices)):
            self.states[i] = self._get_state(self.indices[i] - 1)
            self.new_states[i] = self._get_state(self.indices[i])

        return (np.transpose(self.states, axes=(0, 2, 3, 1)),
                self.actions[self.indices],
                self.rewards[self.indices],
                np.transpose(self.new_states, axes=(0, 2, 3, 1)),
                self.dones[self.indices].astype(np.float32))


class CNN:
    def __init__(self, img_size, frames_per_state, out_size):
        inputs = layers.Input(shape=(img_size, img_size, frames_per_state,))

        layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = layers.Flatten()(layer3)

        layer5 = layers.Dense(512, activation="relu")(layer4)
        action = layers.Dense(out_size, activation="linear")(layer5)

        self.model = keras.Model(inputs=inputs, outputs=action)
        self.loss_fun = keras.losses.Huber()
        self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    def copy_from(self, other):
        self.model.set_weights(other.model.get_weights())

    def predict_one(self, s):
        s = np.array(s).astype(np.float32) / 255.0
        return (self.model(tf.expand_dims(tf.convert_to_tensor(s), 0), training=False)[0]).numpy()

    def predict(self, s):
        s = np.array(s).astype(np.float32) / 255.0
        return self.model.predict(s)

    def train(self, states, target_qs, action_masks):
        states = np.array(states).astype(np.float32) / 255.0
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            all_qs = self.model(states)

            # Apply the masks to the Q-values to get the Q-value for action taken
            qs = tf.reduce_sum(tf.multiply(all_qs, action_masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_fun(target_qs, qs)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()


def get_state_from_frames(state_frames):
    return np.stack(state_frames, axis=2)


def dqn_atari(env_name,
              img_processor,
              mem_capacity=500000,
              min_buf_size=20000,
              frames_per_state=4,
              frame_skip=4,
              train_period=4,
              episodes_num=50000,
              max_steps_num=10000,
              eps_min=0.1,
              explore_steps_num=1000000,
              batch_size=32,
              discount=0.99,
              target_update_period=5000,
              no_op_max=30,
              last_100_av_reward_to_win=np.finfo(np.float32).max):
    # tf.reset_default_graph()
    env = gym.make(env_name)

    img_size = 84
    buffer = ImageMemory(mem_capacity, img_size, img_size, batch_size, frames_per_state)
    steps_counter = 0
    weights_updates = 0

    actions_num = env.action_space.n

    try:

        cnn = CNN(img_size, frames_per_state, actions_num)

        target_cnn = CNN(img_size, frames_per_state, actions_num)
        target_cnn.copy_from(cnn)

        total_rewards = []
        total_clipped_rewards = []
        max_q_values = []
        losses = []

        eps = 1.
        eps_step = (1. - eps_min) / explore_steps_num

        start_time = time.time()
        min_experience_gained = False

        for i_episode in range(episodes_num):
            gc.collect()
            #             print("processing episode", i_episode)
            state_frames = deque(maxlen=frames_per_state)
            total_reward = 0
            total_clipped_reward = 0
            done = False
            q_update_time = 0
            max_q_sum = 0.
            loss_sum = 0.
            train_num = 0

            # initialize state
            observation = env.reset()
            for _ in range(random.randint(1, no_op_max)):
                observation, _, done, _ = env.step(0)
                if done:
                    observation = env.reset()
            frame = img_processor(observation, observation)
            for _ in range(frames_per_state):
                state_frames.append(frame)

            q_vals = cnn.predict_one(get_state_from_frames(state_frames))

            while not done:
                if (not min_experience_gained) or random.random() <= eps:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(q_vals)
                    # action = tf.argmax(cnn.predict_one(get_state_from_frames(state_frames))).numpy()

                # env.render()
                reward = 0.
                for i in range(frame_skip):
                    prev_observation = observation
                    observation, r, done, info = env.step(action)
                    reward += r
                    if done:
                        break

                total_reward += reward
                reward = np.clip(reward, -1, 1)
                total_clipped_reward += reward
                steps_counter += 1

                next_frame = img_processor(prev_observation, observation)
                state_frames.append(next_frame)

                buffer.add_experience(action, next_frame, reward, done)

                if (not min_experience_gained) and buffer.count >= min_buf_size:
                    min_experience_gained = True
                    print("minimum experience gained on episode", i_episode, ". Training started")

                if min_experience_gained:
                    if eps > eps_min:
                        eps -= eps_step

                    q_vals = cnn.predict_one(get_state_from_frames(state_frames))
                    max_q_sum += max(q_vals)
                    q_update_time += 1

                    if steps_counter % train_period == 0:
                        states, actions, rewards, next_states, dones = buffer.get_mini_batch()
                        dones = tf.convert_to_tensor(dones)
                        predicted_next_qs = target_cnn.predict(next_states)
                        target_qs = rewards + discount * tf.reduce_max(predicted_next_qs, axis=1)
                        target_qs = target_qs * (1 - dones) - dones
                        masks = tf.one_hot(actions, actions_num)
                        loss = cnn.train(states, target_qs, masks)
                        loss_sum += loss
                        train_num += 1

                    if steps_counter % target_update_period == 0:
                        target_cnn.copy_from(cnn)

            # calculate stats
            total_rewards.append(total_reward)
            total_clipped_rewards.append(total_clipped_reward)
            weights_updates += train_num
            if q_update_time != 0:
                max_q_values.append(max_q_sum / q_update_time)
            if train_num != 0:
                losses.append(loss_sum / train_num)

            last_100_av_reward = np.array(total_rewards[max(0, i_episode - 100):]).mean()
            last_100_av_clipped_reward = np.array(total_clipped_rewards[max(0, i_episode - 100):]).mean()
            if i_episode % 100 == 0 and i_episode != 0:
                last_100_av_loss = np.array(losses[max(0, len(losses) - 100):]).mean()
                cur_time = time.time()
                print("episode:", i_episode, "total reward:", total_reward, "eps:", eps,
                      "\navg reward/clipped (last 100):", last_100_av_reward, '/', last_100_av_clipped_reward,
                      "\navg loss(last 100):", last_100_av_loss,
                      "\ntime:", cur_time - start_time, "seconds", "total weights updates:", weights_updates)
                print("-------------------------------------------------------------------------")
                start_time = cur_time
            if last_100_av_reward > last_100_av_reward_to_win:
                break

        print("avg reward for last 100 episodes:", np.array(total_rewards[-100:]).mean())
    finally:
        env.close()

    return cnn, total_rewards, max_q_values, losses


def breakout_process(prev_observation, observation):
    return cv2.resize(cv2.cvtColor(np.maximum(prev_observation, observation), cv2.COLOR_RGB2GRAY),
                                  (84, 110), interpolation=cv2.INTER_AREA)[17:101]


def montezuma_process(prev_observation, observation):
    return cv2.resize(cv2.cvtColor(np.maximum(prev_observation, observation), cv2.COLOR_RGB2GRAY)[40:200],
                       (84, 84), interpolation=cv2.INTER_AREA)


def save(path, num, nn, rewards, qs, losses):
    with open(f'{path}/rewards' + str(num), 'w') as f:
        np.savetxt(f, rewards)
    with open(f'{path}/qs' + str(num), 'w') as f:
        np.savetxt(f, qs)
    with open(f'{path}/losses' + str(num), 'w') as f:
        np.savetxt(f, losses)
    nn.model.save(f'{path}/nn' + str(num))


def load_all_stats(path):
    all_rewards = []
    all_qs = []
    all_losses = []
    for agent in range(1, 10):
        with open(f'{path}/rewards' + str(agent), 'r') as f:
            rewards = np.loadtxt(f)
            all_rewards.append(rewards)
        with open(f'{path}/qs' + str(agent), 'r') as f:
            qs = np.loadtxt(f)
            all_qs.append(qs)
        with open(f'{path}/losses' + str(agent), 'r') as f:
            losses = np.loadtxt(f)
            all_losses.append(losses)
    return all_rewards, all_qs, all_losses
