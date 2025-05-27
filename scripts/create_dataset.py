import os
import json
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import torch

# Configuration
image_dir = "./images"
output_json = "captions.json"
model_id = "google/gemma-3-4b-it"
#prompt_text = "Describe this image in detail."
prompt_text = "describe the image with all its objects and their semantic and spatial relation"

# Load model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(model_id, device_map="auto").eval()
processor = AutoProcessor.from_pretrained(model_id)

# Collect image file paths
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Store results
results = {}

# Generate captions
for img_file in image_files:
    try:
        image_path = os.path.join(image_dir, img_file)
        image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            output_ids = generation[0][input_len:]

        caption = processor.decode(output_ids, skip_special_tokens=True)
        results[img_file] = caption

        print(f"[✓] {img_file}: {caption}")

    except Exception as e:
        print(f"[✗] Failed on {img_file}: {e}")

# Save to JSON
with open(output_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved results to {output_json}")

