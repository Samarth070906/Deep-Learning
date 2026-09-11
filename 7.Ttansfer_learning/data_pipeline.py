from datasets import load_dataset, DownloadConfig
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


# ============================================================
# CONFIGURATION
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

# Number of images per class
TRAIN_IMAGES_PER_CLASS = 100
VAL_IMAGES_PER_CLASS = 25

# Image and training settings
IMAGE_SIZE = 224
BATCH_SIZE = 8


# ============================================================
# HUGGING FACE DOWNLOAD CONFIGURATION
# ============================================================

# Increase retry attempts in case the internet connection
# to Hugging Face is temporarily interrupted.
download_config = DownloadConfig(
    max_retries=10
)


# ============================================================
# CREATE DATA STREAMS
# ============================================================

print("=" * 60)
print("Loading Food-101 dataset...")
print("=" * 60)

train_stream = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
    download_config=download_config
)

val_stream = load_dataset(
    DATASET_NAME,
    split="validation",
    streaming=True,
    download_config=download_config
)


# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = train_stream.features["label"].names

print("\nTotal Food-101 classes:", len(class_names))


# Convert original Food-101 labels into our new labels 0-9
selected_label_ids = {
    class_names.index(class_name): new_label
    for new_label, class_name in enumerate(SELECTED_CLASSES)
}


print("\nSelected classes:")
for i, class_name in enumerate(SELECTED_CLASSES):
    print(f"{i}: {class_name}")


# ============================================================
# FUNCTION TO COLLECT IMAGES
# ============================================================

def collect_images(stream, images_per_class):
    """
    Collect a fixed number of images from each selected class.
    """

    collected = []

    class_counts = defaultdict(int)

    print("\nCollecting images...")

    for sample in stream:

        original_label = sample["label"]

        # Ignore classes that we don't need
        if original_label not in selected_label_ids:
            continue

        new_label = selected_label_ids[original_label]

        # Skip if we already have enough images
        if class_counts[new_label] >= images_per_class:
            continue

        try:
            # Convert image to RGB
            image = sample["image"].convert("RGB")

            collected.append((image, new_label))

            class_counts[new_label] += 1

        except Exception as error:

            print(
                f"Skipping corrupted image "
                f"from class {SELECTED_CLASSES[new_label]}: {error}"
            )

            continue

        # Stop when every class has enough images
        if all(
            class_counts[i] >= images_per_class
            for i in range(len(SELECTED_CLASSES))
        ):
            break

    print("\nImages collected:", len(collected))

    print("\nClass distribution:")

    for i, class_name in enumerate(SELECTED_CLASSES):
        print(
            f"{class_name}: "
            f"{class_counts[i]} images"
        )

    return collected


# ============================================================
# COLLECT TRAINING DATA
# ============================================================

print("\n" + "=" * 60)
print("COLLECTING TRAINING DATA")
print("=" * 60)

train_data = collect_images(
    train_stream,
    TRAIN_IMAGES_PER_CLASS
)


# ============================================================
# COLLECT VALIDATION DATA
# ============================================================

print("\n" + "=" * 60)
print("COLLECTING VALIDATION DATA")
print("=" * 60)

val_data = collect_images(
    val_stream,
    VAL_IMAGES_PER_CLASS
)


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

# ImageNet preprocessing because our models
# were pretrained on ImageNet.

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CUSTOM PYTORCH DATASET
# ============================================================

class FoodDataset(Dataset):

    def __init__(self, data, transform=None):

        self.data = data
        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        image, label = self.data[index]

        if self.transform:

            image = self.transform(image)

        return image, label


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = FoodDataset(
    train_data,
    transform
)

val_dataset = FoodDataset(
    val_data,
    transform
)


# ============================================================
# CHECK GPU
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# DATA PIPELINE INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA PIPELINE READY")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Number of classes:",
    len(SELECTED_CLASSES)
)

print(
    "Image size:",
    f"{IMAGE_SIZE} x {IMAGE_SIZE}"
)

print(
    "Batch size:",
    BATCH_SIZE
)


# ============================================================
# TEST ONE BATCH
# ============================================================

images, labels = next(
    iter(train_loader)
)

print("\n" + "=" * 60)
print("BATCH TEST")
print("=" * 60)

print(
    "Image tensor shape:",
    images.shape
)

print(
    "Labels shape:",
    labels.shape
)


# Move batch to GPU

images = images.to(
    device,
    non_blocking=True
)

labels = labels.to(
    device,
    non_blocking=True
)


print(
    "Images device:",
    images.device
)

print(
    "Labels device:",
    labels.device
)


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SUCCESS: DATA PIPELINE IS WORKING")
print("=" * 60)

print(
    "Ready for transfer learning with:"
)

print("1. AlexNet")
print("2. VGG16")
print("3. ResNet50")
print("4. EfficientNetB0")

print("=" * 60)