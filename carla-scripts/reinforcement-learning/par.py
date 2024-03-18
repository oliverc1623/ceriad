#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from CarlaEnv import CustomEnv
import matplotlib.pyplot as plt
import numpy as np


# ## Actor

# In[2]:


# load model and env
model = PPO.load("ppo/carla-lanefollow-empty_trial1")
vec_env = CustomEnv(ego_vehicle='car1')


# ## Planner

# In[1]:


from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# OpenAI ChatCompletions API client initialization
client = OpenAI(
    api_key=""
)


# In[3]:


def selectAction(prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "Your goal is to drive safely and efficiently. You are directing the ego vehicle in our simulation, selecting actions when prompted. Respond with the action name only, and nothing else."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()


# In[4]:


selectAction("There is a car right in front of me.")


# ## Reporter

# In[1]:


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


# In[2]:





# ## Actor-Reporter-Planner

# In[4]:


obs, info = env.reset()


# In[ ]:


# TODO:
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, truncated, info = vec_env.step(action)
#     if dones:
#         obs, info = vec_env.reset()

