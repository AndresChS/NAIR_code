import myosuite
from myosuite.utils import gym
import skvideo.io
import numpy as np
import os
from IPython.display import HTML
from base64 import b64encode
 
def show_video(video_path, video_width = 400):
   
  video_file = open(video_path, "r+b").read()
 
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")

env = gym.make('myoElbowPose1D6MRandom-v0')

env.reset();
import warnings
warnings.filterwarnings('ignore')

from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.utils.train_agent import train_agent
import myosuite
    
policy_size = (32, 32)
vf_hidden_size = (128, 128)
seed = 123
rl_step_size = 0.1
e = GymEnv(env)

policy = MLP(e.spec, hidden_sizes=policy_size, seed=seed, init_log_std=-0.25, min_log_std=-1.0)

baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=64, hidden_sizes=vf_hidden_size, \
                    epochs=2, learn_rate=1e-3)

agent = NPG(e, policy, baseline, normalized_step_size=rl_step_size, \
            seed=seed, save_logs=True)

print("========================================")
print("Starting policy learning")
print("========================================")

train_agent(job_name='.',
            agent=agent,
            seed=seed,
            niter=200,
            gamma=0.995,
            gae_lambda=0.97,
            num_cpu=8,
            sample_mode="trajectories",
            num_traj=96,
            num_samples=0,
            save_freq=100,
            evaluation_rollouts=10)

print("========================================")
print("Job Finished.") 
print("========================================")

policy = "iterations/best_policy.pickle"

import pickle
pi = pickle.load(open(policy, 'rb'))

AngleSequence = [60, 30, 30, 60, 80, 80, 60, 30, 80, 30, 80, 60]
env.reset()
frames = []
for ep in range(len(AngleSequence)):
    print("Ep {} of {} testing angle {}".format(ep, len(AngleSequence), AngleSequence[ep]))
    env.env.target_jnt_value = [np.deg2rad(AngleSequence[int(ep)])]
    env.env.target_type = 'fixed'
    env.env.weight_range=(0,0)
    env.env.update_target()
    for _ in range(40):
        frame = env.sim.render(width=400, height=400,mode='offscreen', camera_name=None)
        frames.append(frame[::-1,:,:])
        o = env.get_obs()
        a = pi.get_action(o)[0]
        next_o, r, done, ifo = env.step(a) # take an action based on the current observation
env.close()

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/arm.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
show_video('videos/arm.mp4')
