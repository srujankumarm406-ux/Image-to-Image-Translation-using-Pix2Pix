import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import requests
from io import BytesIO

def image_to_image():
    print("Loading Image-to-Image model...")

    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id)
    pipe = pipe.to("cpu")  # change to "cuda" if GPU available

    # Load sample image from URL
    url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
    response = requests.get(url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((512, 512))

    prompt = input("Enter transformation prompt: ")

    print("Generating transformed image...")
    image = pipe(prompt=prompt, image=init_image, strength=0.75).images[0]

    image.save("output_task4.png")

    print("Image saved as output_task4.png")

if __name__ == "__main__":
    image_to_image()task1_gpt2.py
