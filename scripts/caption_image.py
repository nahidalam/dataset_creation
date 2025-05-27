from transformers import pipeline
from PIL import Image

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-4b-it",
    device="cuda",
    torch_dtype="bfloat16"  # or "float16" if bfloat16 unsupported
)

image = Image.open("example.jpg").convert("RGB")
prompt = "describe the image with its objects and their relation and relative location"

result = pipe(image, prompt)
print(result[0]["generated_text"])

