#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from CarlaEnv import CustomEnv
from HardEnv import HardEnv
import matplotlib.pyplot as plt
import numpy as np


# ## Actor-Reporter-Planner

# ## Actor

# In[2]:


# load model and env
model = PPO.load("ppo/carla-lanefollow-empty_trial1")
vec_env = HardEnv(ego_vehicle='car1')


# ## Planner

# In[3]:


from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# OpenAI ChatCompletions API client initialization
client = OpenAI(
    api_key=""
)


# In[5]:


def selectAction(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Your goal is to drive safely and efficiently. You are directing the ego vehicle in our simulation, selecting actions when prompted."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()


# In[6]:


selectAction("There is a motocycle directly in front of me that appears to be stoppping. Select 1) brake or 2) continue for me.")


# ## Reporter

# In[7]:


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


# In[10]:


vec_env = CustomEnv(ego_vehicle='car1')
obs, info = vec_env.reset()

plt.imsave("basic.jpg", obs['image'][0],cmap='gray')

# get initial llava report
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are objects worth noting in the current scenario? What are the vehicles in the image, their positions, and are they a possible threat to the ego vehicle based on their driving? Answer as the point of view of the driver in the image. I am driving at 40 mph. Be as succinct as possible."
image_file = "basic.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()
eval_model(args)


# In[16]:


report = "In the image, there are several people and bicycles on the sidewalk, which might be a distraction for the driver. Additionally, there are two cars in the scene, one behind the ego vehicle and another further ahead. The cars are not a direct threat to the ego vehicle, but the driver should remain cautious and maintain a safe distance from the cars to avoid any potential accidents."
report += " Select 1) brake or 2) continue for me."
selectAction(report)


# Vehicle obstacle test:
# Trial 1: pass
# Trial 2: pass
# Trial 3: fail
# Trial 4: fail
# Trial 5: fail
# Trial 6: pass
# Trial 7: pass
# Trial 8: fail
# Trial 9: fail
# Trial 10: pass

# Vehicle lane follow test, zero-shot:
# Trial 1: fail
# Trial 2: pass
# Trial 3: pass
# Trial 4: pass
# Trial 5: pass
# Trial 6: pass
# Trial 7: fail
# Trial 8: pass
# Trial 9: fail
# Trial 10: pass
