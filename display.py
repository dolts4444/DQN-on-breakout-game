
# coding: utf-8

# In[1]:


import base64
from collections import deque
import os
import pathlib
import shutil

from IPython import display as ipydisplay
import torch

from utils_env import MyEnv
from utils_drl import Agent


# In[2]:


target = 78
model_name = f"model_{target:03d}"
model_path = f"./models/{model_name}"
device = torch.device("cpu")
env = MyEnv(device)
agent = Agent(env.get_action_dim(), device, 0.99, 0, 0, 0, 1, model_path, 
             use_dueling = True,use_PR = True,use_DDQN = True)


# In[3]:


obs_queue = deque(maxlen=5)
avg_reward, frames = env.evaluate(obs_queue, agent, render=True)
print(f"Avg. Reward: {avg_reward:.1f}")

get_ipython().system('rm -r eval_*')
target_dir = f"eval_{target:03d}"
os.mkdir(target_dir)
for ind, frame in enumerate(frames):
    frame.save(os.path.join(target_dir, f"{ind:06d}.png"), format="png")


# In[4]:


path_to_mp4 = os.path.join(target_dir, "movie.mp4")


# In[5]:


if not os.path.exists(path_to_mp4):
    shutil.move(target_dir, "tmp_eval_frames")
    # Generate an mp4 video from the frames
    get_ipython().system('ffmpeg -i "./tmp_eval_frames/%06d.png" -pix_fmt yuv420p -y ./tmp_eval_movie.mp4 > /dev/null 2>&1')
    get_ipython().system('rm -r tmp_eval_frames')
    os.mkdir(target_dir)
    shutil.move("tmp_eval_movie.mp4", path_to_mp4)


# In[6]:


HTML_TEMPLATE = """<video alt="{alt}" autoplay loop controls style="height: 400px;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4" />
</video>"""

def show_video(path_to_mp4: str) -> None:
    """show_video creates an HTML element to display the given mp4 video in IPython."""
    mp4 = pathlib.Path(path_to_mp4)
    video_b64 = base64.b64encode(mp4.read_bytes())
    html = HTML_TEMPLATE.format(alt=mp4, data=video_b64.decode('ascii'))
    ipydisplay.display(ipydisplay.HTML(data=html))


# In[7]:


show_video(path_to_mp4)

