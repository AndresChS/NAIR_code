import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

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
        if self.viewer is None:
            self.viewer = DoublePendulumViewer()
        self.viewer.update(self.state)
        return self.viewer.render(mode)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class DoublePendulumViewer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.pendulum1, = self.ax.plot([], [], 'o-', lw=2)
        self.pendulum2, = self.ax.plot([], [], 'o-', lw=2)
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_aspect('equal')
        self.ax.grid()
        self.time_template = 'time = %.1fs'
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        self.line1 = [[0, 0], [0, 0]]
        self.line2 = [[0, 0], [0, 0]]

    def update(self, state):
        th1, _, th2, _ = state
        l1 = 1.0
        l2 = 1.0
        x1 = l1 * np.sin(th1)
        y1 = -l1 * np.cos(th1)
        x2 = x1 + l2 * np.sin(th2)
        y2 = y1 - l2 * np.cos(th2)

        self.pendulum1.set_data([0, x1], [0, y1])
        self.pendulum2.set_data([x1, x2], [y1, y2])

        self.line1[0][0] = 0
        self.line1[1][0] = x1
        self.line1[0][1] = 0
        self.line1[1][1] = y1

        self.line2[0][0] = x1
        self.line2[1][0] = x2
        self.line2[0][1] = y1
        self.line2[1][1] = y2

        self.time_text.set_text(self.time_template % 0)
        return self.pendulum1, self.pendulum2, self.time_text

    def render(self, mode='human'):
        plt.pause(0.001)
        return self.fig

    def close(self):
        plt.close()

env = DoublePendulumEnv()
observation = env.reset()

# Crear la animación
fig, ax = plt.subplots()
pendulum1, = ax.plot([], [], 'o-', lw=2)
pendulum2, = ax.plot([], [], 'o-', lw=2)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid()
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
line1 = [[0, 0], [0, 0]]
line2 = [[0, 0], [0, 0]]

def update(frame):
    observation, _, _, _ = env.step(env.action_space.sample())
    th1, _, th2, _ = observation
    l1 = 1.0
    l2 = 1.0
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th2)
    y2 = y1 - l2 * np.cos(th2)

    pendulum1.set_data([0, x1], [0, y1])
    pendulum2.set_data([x1, x2], [y1, y2])

    line1[0][0] = 0
    line1[1][0] = x1
    line1[0][1] = 0
    line1[1][1] = y1

    line2[0][0] = x1
    line2[1][0] = x2
    line2[0][1] = y1
    line2[1][1] = y2

    time_text.set_text(time_template % (frame * env.dt))
    return pendulum1, pendulum2, time_text

ani = FuncAnimation(fig, update, frames=range(100), interval=50)

# Obtener la imagen de la figura como arreglo
fig.canvas.draw()
image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
print("Tamaño del arreglo de la imagen RGB:", len(image))

plt.show()
env.close()
"""
# Guardar la animación en un archivo de video MP4 usando imageio
filename = "double_pendulum.mp4"
with imageio.get_writer(filename, fps=15) as writer:
    for i in range(100):
        update(i)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        print("Tamaño del arreglo:", image.size)
        print("Dimensiones de la imagen original:", fig.canvas.get_width_height())
        image = image.reshape((480, 640, 3))
        writer.append_data(image)

plt.show()
env.close()
"""