from datasets import load_dataset

# ============================================================
# Food-101 Dataset Configuration
# ============================================================

DATASET_NAME = "ethz/food101"

SELECTED_CLASSES = [
    "apple_pie",
    "cheesecake",
    "chicken_curry",
    "french_fries",
    "fried_rice",
    "hamburger",
    "ice_cream",
    "pizza",
    "sushi",
    "tacos"
]

TRAIN_IMAGES_PER_CLASS = 100
VAL_IMAGES_PER_CLASS = 25


# ============================================================
# Load Food-101 using streaming
# ============================================================

train_dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

val_dataset = load_dataset(
    DATASET_NAME,
    split="validation",
    streaming=True
)


# ============================================================
# Get original Food-101 class names
# ============================================================

class_names = train_dataset.features["label"].names


# ============================================================
# Convert class names to original label IDs
# ============================================================

selected_label_ids = {
    class_names.index(class_name): new_label
    for new_label, class_name in enumerate(SELECTED_CLASSES)
}


# ============================================================
# Display configuration
# ============================================================

print("=" * 50)
print("Food-101 Transfer Learning Dataset")
print("=" * 50)

print("Dataset:", DATASET_NAME)
print("Total Food-101 classes:", len(class_names))

print("\nSelected classes:")

for new_label, class_name in enumerate(SELECTED_CLASSES):
    original_label = class_names.index(class_name)

    print(
        f"{new_label}: {class_name} "
        f"(original label {original_label})"
    )

print("\nTraining images per class:", TRAIN_IMAGES_PER_CLASS)
print("Validation images per class:", VAL_IMAGES_PER_CLASS)
print("Total training images:", TRAIN_IMAGES_PER_CLASS * len(SELECTED_CLASSES))
print("Total validation images:", VAL_IMAGES_PER_CLASS * len(SELECTED_CLASSES))