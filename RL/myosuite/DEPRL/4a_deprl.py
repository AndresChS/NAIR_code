import myosuite
import deprl
from myosuite.utils import gym

T = 1000 # length of episode
env = gym.make('myoLegWalk-v0')
obs = env.reset()

policy = deprl.load_baseline(env)

obs = env.reset()
for _ in range(T):
    action = policy(obs[0])
    obs, rew, done, info = env.step(action)
    if done:
        break
env.close()
print('Done!')

