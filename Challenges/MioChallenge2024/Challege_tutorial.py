from myosuite.utils import gym
# Include the locomotion track environment, uncomment to select the manipulation challenge
env = gym.make('myoChallengeRunTrackP1-v0')
#env = gym.make('myoChallengeBimanual-v0')


env.reset()

# Repeat 1000 time steps
for _ in range(1000):

    # Activate mujoco rendering window
    env.mj_render()


    # Get observation from the envrionment, details are described in the above docs
    obs = env.get_obs()
    #current_time = obs['time']
    #print(current_time)


    # Take random actions
    action = env.action_space.sample()


    # Environment provides feedback on action
    next_obs, reward, terminated, truncated, info = env.step(action)


    # Reset training if env is terminated
    if terminated:
        next_obs, info = env.reset()