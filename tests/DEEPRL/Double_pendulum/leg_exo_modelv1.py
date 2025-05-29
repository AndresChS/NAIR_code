import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pyglet
from pyglet import shapes
from pyglet.gl import *

class DoublePendulumEnv(gym.Env):
    def __init__(self):
        self.gravity = 9.8
        self.dt = 0.05
        self.max_torque = 2.0
        self.max_speed = 8.0

        high = np.array([np.pi, self.max_speed, np.pi, self.max_speed])
        self.action_space = spaces.Box(low=np.array([-self.max_torque, -self.max_torque]), 
                                       high=np.array([self.max_torque, self.max_torque]), 
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.state = None
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        th1, thdot1, th2, thdot2 = self.state
        g = self.gravity
        m1 = 1.0
        m2 = 1.0
        l1 = 1.0
        l2 = 1.0
        dt = self.dt
        torque1, torque2 = action

        # Equations of motion
        newthdot1 = thdot1 + (-3 * g / (2 * l1) * np.sin(th1 + np.pi) +
                              3. / (m1 * l1 ** 2) * torque1 - 3. / (m1 * l1 ** 2) * np.cos(th1 - th2) *
                              (g * np.sin(th1 - np.pi) / l2 - np.sin(th1 - th2) * thdot2 ** 2 * l1 -
                               np.cos(th1 - th2) * thdot1 ** 2 * l1) /
                              (3 - np.cos(2 * th1 - 2 * th2))) * dt
        newthdot2 = thdot2 + (-3 * g / (2 * l2) * np.sin(th2 + np.pi) +
                              3. / (m2 * l2 ** 2) * torque2 * np.cos(th1 - th2) - 3. / (m2 * l2 ** 2) *
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
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = 2.4 * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 6.0
        polelen = scale * (2 * 1.0)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            self.viewer = pyglet.window.Window(screen_width, screen_height, resizable=False)
            self.viewer.set_caption("Double Pendulum")
            self.pendulum1 = shapes.Line(0, 0, 0, 0, width=polewidth, color=(255, 0, 0))
            self.pendulum2 = shapes.Line(0, 0, 0, 0, width=polewidth, color=(0, 255, 0))

        self.viewer.clear()
        self.viewer.switch_to()
        self.viewer.dispatch_events()

        cartx = screen_width / 2.0  # MIDDLE OF CART
        carty = screen_height / 2.0  # MIDDLE OF SCREEN

        self.pendulum1.x = cartx
        self.pendulum1.y = carty
        self.pendulum1.x2 = cartx + np.sin(self.state[0]) * polelen
        self.pendulum1.y2 = carty - np.cos(self.state[0]) * polelen

        self.pendulum2.x = self.pendulum1.x2
        self.pendulum2.y = self.pendulum1.y2
        self.pendulum2.x2 = self.pendulum1.x2 + np.sin(self.state[2]) * polelen
        self.pendulum2.y2 = self.pendulum1.y2 - np.cos(self.state[2]) * polelen

        self.pendulum1.draw()
        self.pendulum2.draw()

        self.viewer.flip()

        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

env = DoublePendulumEnv()
observation = env.reset()

# Ejemplo de cómo tomar una acción en el entorno
action = np.array([1.0, -0.5])  # Torque para la primera barra, y torque inverso para la segunda
next_state, reward, done, _ = env.step(action)

# Renderizar el entorno
env.render()

# Mantener la ventana abierta
pyglet.app.run()
