import json

# Path to the generated captions file
captions_file = "captions.json"

# Load the JSON
with open(captions_file, "r") as f:
    captions = json.load(f)

# Pretty-print the JSON structure
print(json.dumps(captions, indent=2))

# Optional: If you want to use the dict for further processing
for filename, caption in captions.items():
    print(f"{filename}: {caption}")

