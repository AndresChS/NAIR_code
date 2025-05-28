
from myosuite.utils import gym
import numpy as np
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    env_id = "ExoLeg40MuscFlexoExtEnv-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    #vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)
    env_mj = gym.make(env_id)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

    obs = env.reset()
    obs_mj = env_mj.reset()
    actions = []
    print(model)
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        actions.append(action[0])

    for i in actions:
     
        env_mj.step(np.array(i))
        time.sleep(0.01)
        env_mj.mj_render()
    
    env_mj.close()
