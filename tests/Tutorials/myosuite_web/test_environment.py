#************************* test environment **************************
"""
import myosuite
import gym
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
env.close()
"""

#************************ Activate and visualize finger movements ********
"""
import myosuite
import gym
env = gym.make('myoHandPoseRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
env.close()
"""
#************************ Test trained policy **************************

import myosuite
import gym
policy = "best_policy.pickle"

import pickle
pi = pickle.load(open(policy, 'rb'))
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action

#************************ Test muscle fatigue **************************
"""
import myosuite 
import gym
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action

# Add muscle fatigue
env = gym.make('myoFatiElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
env.close()
"""
#************************ Test Sarcopenia **************************
"""
import myosuite
import gym
env = gym.make('myoElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action

# Add muscle weakness
env = gym.make('myoSarcElbowPose1D6MRandom-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
env.close()
"""
#************************ Test Physical tendon transfer **************************
"""
import myosuite
import gym
env = gym.make('myoHandKeyTurnFixed-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action

# Add tendon transfer
env = gym.make('myoTTHandKeyTurnFixed-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
env.close()
"""
# python3 hydra_mjrl_launcher.py --config-path config --config-name hydra_biomechanics_config.yaml 
#    hydra/output=local hydra/launcher=local env=myoHandPoseRandom-v0 job_name=[Absolute Path of the policy] 
#    rl_num_iter=[New Total number of iterations]


#************************ Load DEP_RL Baseline **************************
"""
import myosuite
import gym
import deprl

# we can pass arguments to the environments here
env = gym.make('myoLegWalk-v0', reset_type='random')
policy = deprl.load_baseline(env)
obs = env.reset()
for i in range(1000):
    env.mj_render()
    action = policy(obs)
    obs, *_ = env.step(action)
env.close()
"""