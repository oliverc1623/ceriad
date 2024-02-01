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

while 1:
    prompt = input("USER: <image>\n")
    prompt += "\nAssistant:"
    max_new_tokens = 200
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    print(outputs[0])