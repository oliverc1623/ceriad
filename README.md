# 🚗 Collaborative Embodied Reasoning in Autonomous Driving

---

An extension of the Planner-Actor-Reporter framework applied to autonomous vehicles in Highway-Env and CARLA.

## Contents

- [Install](#install)
- [Planner](#planner)
- [Actor](#actor)
- [Reporter](#reporter)

## Install

If you only want to use Highway-env run the pip install commands:
1. 
```
pip install highway-env
pip install stable-baselines3[extra]
```

For CARLA, please follow the Nautilus GUI setup [guide](nautilus-files/README.md).

## Planner

 The planner is designed to take in prompts from the user and the environment inference from the reporter and produce an appropriate action for the actor, which is the reinforcement learning agent. We use pre-trained large language models as the planner. In this case, we have used GPT-3.5-turbo as our planner, utilizing the OpenAI's API. We have modeled the API output to only give the appropriate safe action for our vehicle.

### ChatGPT

We leverage OpenAI's API to use ChatGPT-3.5-Turbo. 

## Actor

The Actor component of this framework adaptation serves two purposes: (i) give fine-control commands to the ego vehicle; and (ii) send images and actions to the reporter. The Actor for both Highway-Env and CARLA are trained using DRL algorithms. In future work, we hope to apply multi-goal reinforcement learning so that the Planner can issue more high-level commands for the Actor. 

### Highway-Env

### CARLA

## Reporter

### Hard-coded Reporter

### LLaVA  