# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:17:17 2024

@author: andre
"""

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class DoublePendulumEnv(gym.Env):
    def __init__(self):
        self.gravity = 9.8
        self.dt = 0.05
        self.max_torque = 2.0
        self.max_speed = 8.0
        self.viewer = None

        high = np.array([np.pi, self.max_speed, np.pi, self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.state = None
        self.viewer = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th1, thdot1, th2, thdot2 = self.state
        g = self.gravity
        m1 = 1.0
        m2 = 1.0
        l1 = 1.0
        l2 = 1.0
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u

        # Equations of motion
        # Modify equations here to include the second pendulum
        newthdot1 = thdot1 + (-3 * g / (2 * l1) * np.sin(th1 + np.pi) +
                              3. / (m1 * l1 ** 2) * u - 3. / (m1 * l1 ** 2) * np.cos(th1 - th2) *
                              (g * np.sin(th1 - np.pi) / l2 - np.sin(th1 - th2) * thdot2 ** 2 * l1 -
                               np.cos(th1 - th2) * thdot1 ** 2 * l1) /
                              (3 - np.cos(2 * th1 - 2 * th2))) * dt
        newthdot2 = thdot2 + (-3 * g / (2 * l2) * np.sin(th2 + np.pi) +
                              3. / (m2 * l2 ** 2) * u * np.cos(th1 - th2) - 3. / (m2 * l2 ** 2) *
                              (g * np.sin(th2 - np.pi) / l1 + np.sin(th1 - th2) * thdot1 ** 2 * l1 -
                               np.cos(th1 - th2) * thdot2 ** 2 * l1) /
                              (3 - np.cos(2 * th1 - 2 * th2))) * dt
        newth1 = th1 + newthdot1 * dt
        newth2 = th2 + newthdot2 * dt
        newthdot1 = np.clip(newthdot1, -self.max_speed, self.max_speed)
        newthdot2 = np.clip(newthdot2, -self.max_speed, self.max_speed)

        self.state = np.array([newth1, newthdot1, newth2, newthdot2])
        return np.array(self.state), 0, False, {}

    def reset(self):
        high = np.array([np.pi, self.max_speed, np.pi, self.max_speed])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.steps_beyond_done = None
        return np.array(self.state)
    
    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

env = DoublePendulumEnv()
observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()