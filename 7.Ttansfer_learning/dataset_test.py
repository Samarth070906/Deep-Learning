from datasets import load_dataset

# Hugging Face Food-101 dataset
dataset_url = "ethz/food101"

# Load dataset using streaming
dataset = load_dataset(
    dataset_url,
    split="train",
    streaming=True
)

# Get class names
class_names = dataset.features["label"].names

print("Number of classes:", len(class_names))
print("\nFood-101 Classes:\n")

for i, name in enumerate(class_names):
    print(f"{i:3d} : {name}")