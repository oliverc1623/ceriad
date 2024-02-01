import requests
from PIL import Image

image = Image.open("../frames/frame_000.png")

import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

from transformers import pipeline

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline(model=model_id, model_kwargs={"quantization_config": quantization_config})

max_new_tokens = 200
prompt = "USER: <image>\nDescribe this driving scene. I am the yellow car. My throttle speed is 0.00930562149733305 and steering angle is 0.2912810146808624. My coordinate is (100.0, 3.999999910593033).\nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})