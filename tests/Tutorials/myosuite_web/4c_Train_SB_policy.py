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

from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=0)

print("========================================")
print("Starting policy learning")
print("========================================")

model.learn(total_timesteps=1000)

print("========================================")
print("Job Finished.") 
print("========================================")

model.save('ElbowPose_policy')

policy = "ElbowPose_policy.zip"

pi = PPO.load(policy)

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
        frame = env.sim.renderer.render_offscreen(width=400, height=400)
        frames.append(frame[::-1,:,:])
        o = env.get_obs()
        a = pi.predict(o)[0]
        next_o, r, done, ifo = env.step(a) # take an action based on the current observation
env.close()

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/arm.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
show_video('videos/arm.mp4')