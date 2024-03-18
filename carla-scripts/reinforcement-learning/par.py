#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gymnasium as gym
import torch
import time
from stable_baselines3 import PPO
from CarlaEnv import CustomEnv
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
import warnings
warnings.filterwarnings('ignore')


# ## Actor

# In[5]:


# load model and env
model = PPO.load("ppo/carla-lanefollow-empty_trial1")
vec_env = CustomEnv(ego_vehicle='car1')


# ## Planner

# In[4]:


# OpenAI ChatCompletions API client initialization
client = OpenAI(
    api_key="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"
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


# In[7]:


selectAction("There is a car right in front of me.")


# ## Reporter

# In[1]:


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model


# In[2]:


model_path = "liuhaotian/llava-v1.5-7b"
prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

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


# In[ ]:




