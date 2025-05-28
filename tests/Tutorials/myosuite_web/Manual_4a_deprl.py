# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:15:40 2023

@author: andre
"""
import myosuite
from myosuite.utils import gym
import deprl
import skvideo.io
import numpy as np
import os
import matplotlib.pyplot as plt

from IPython.display import HTML
from base64 import b64encode

def show_video(video_path, video_width = 400):
   
  video_file = open(video_path, "r+b").read()
 
  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")
 
env = gym.make('myoElbowPose1D6MRandom-v0')
print('List of cameras available', [env.sim.model.camera(i).name for i in range(env.sim.model.ncam)]) 

T = 10000 # length of episode
env = gym.make('myoLegWalk-v0')
obs = env.reset()

policy = deprl.load_baseline(env)

obs = env.reset()
frames = []
for _ in range(T):
    action = policy(obs)
    frame = env.sim.renderer.render_offscreen(
                        width=400,
                        height=400,
                        camera_id=0)
    frames.append(frame)
    obs, rew, done, info = env.step(action)
    
    if done:
        break
env.close()
print('Done!')

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/temp2.mp4', np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})

# show in the notebook
show_video('videos/temp.mp4')


plt.imshow(frame)