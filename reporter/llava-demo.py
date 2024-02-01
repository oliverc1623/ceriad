import argparse
import requests
from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline


def main(args):
    image = Image.open(args.filename)
    print(image)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    model_id = "llava-hf/llava-1.5-7b-hf"

    pipe = pipeline(
        model=model_id, model_kwargs={"quantization_config": quantization_config}
    )

    while 1:
        prompt = input("Prompt: \n")
        prompt = "USER: <image>\n" + prompt + "\nAssistant:"
        max_new_tokens = 200
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        print(outputs[0]["generated_text"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="LLaVA Inference Demo",
        description="Runs LLaVA",
    )
    parser.add_argument("-f", "--filename", type=str)
    args = parser.parse_args()
    main(args)
