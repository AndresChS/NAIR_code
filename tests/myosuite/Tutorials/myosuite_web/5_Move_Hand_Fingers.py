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
  return HTML(f"""""")

import mujoco
env = gym.make('myoHandPoseRandom-v0', normalize_act = False)

env.env.init_qpos[:] = np.zeros(len(env.env.init_qpos),)
mjcModel = env.env.sim.model

# print("Muscles:")
# for i in range(mjcModel.na):
#     print([i,mjcModel.actuator(i).name])

# print("\nJoints:")
# for i in range(mjcModel.njnt):
#     print([i,mjcModel.joint(i).name])


musc_fe = [mjcModel.actuator('FDP2').id,mjcModel.actuator('EDC2').id]
L_range = round(1/mjcModel.opt.timestep)
skip_frame = 50
env.reset()

frames_sim = []
for iter_n in range(3):
    print("iteration: "+str(iter_n))
    res_sim = []
    for rp in range(2): #alternate between flexor and extensor
        for s in range(L_range):
            if not(s%skip_frame):
                frame = env.sim.renderer.render_offscreen(
                                width=400,
                                height=400,
                                camera_id=3)
                frames_sim.append(frame)
            
            ctrl = np.zeros(mjcModel.na,)

            act_val = 1 # maximum muscle activation
            if rp==0:
                ctrl[musc_fe[0]] = act_val
                ctrl[musc_fe[1]] = 0
            else:
                ctrl[musc_fe[1]] = act_val
                ctrl[musc_fe[0]] = 0                        
            env.step(ctrl)

os.makedirs('videos', exist_ok=True)
# make a local copy
skvideo.io.vwrite('videos/MyoSuite1.mp4', np.asarray(frames_sim),outputdict={"-pix_fmt": "yuv420p"})

# show in the notebook
show_video('videos/MyoSuite.mp4')