import torch
import torch.nn as nn

from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    vgg16,
    VGG16_Weights,
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FUNCTION 1: ALEXNET
# ============================================================

def create_alexnet():

    print("\nLoading pretrained AlexNet...")

    # Load AlexNet pretrained on ImageNet
    model = alexnet(
        weights=AlexNet_Weights.DEFAULT
    )

    # Freeze all pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the final classification layer
    # Original output = 1000 ImageNet classes
    # New output = 10 Food-101 selected classes
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# FUNCTION 2: VGG16
# ============================================================

def create_vgg16():

    print("\nLoading pretrained VGG16...")

    # Load VGG16 pretrained on ImageNet
    model = vgg16(
        weights=VGG16_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace final classification layer
    model.classifier[6] = nn.Linear(
        model.classifier[6].in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# FUNCTION 3: RESNET50
# ============================================================

def create_resnet50():

    print("\nLoading pretrained ResNet50...")

    # Load ResNet50 pretrained on ImageNet
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace final fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# FUNCTION 4: EFFICIENTNET-B0
# ============================================================

def create_efficientnet_b0():

    print("\nLoading pretrained EfficientNet-B0...")

    # Load EfficientNet-B0 pretrained on ImageNet
    model = efficientnet_b0(
        weights=EfficientNet_B0_Weights.DEFAULT
    )

    # Freeze pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace final classification layer
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )

    model = model.to(DEVICE)

    return model


# ============================================================
# FUNCTION TO COUNT PARAMETERS
# ============================================================

def count_parameters(model):

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


# ============================================================
# TEST ALL FOUR MODELS
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TRANSFER LEARNING MODEL TEST")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # AlexNet
    # --------------------------------------------------------

    alexnet_model = create_alexnet()

    total, trainable = count_parameters(
        alexnet_model
    )

    print("\nAlexNet")
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")


    # --------------------------------------------------------
    # VGG16
    # --------------------------------------------------------

    vgg16_model = create_vgg16()

    total, trainable = count_parameters(
        vgg16_model
    )

    print("\nVGG16")
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")


    # --------------------------------------------------------
    # ResNet50
    # --------------------------------------------------------

    resnet50_model = create_resnet50()

    total, trainable = count_parameters(
        resnet50_model
    )

    print("\nResNet50")
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")


    # --------------------------------------------------------
    # EfficientNet-B0
    # --------------------------------------------------------

    efficientnet_model = create_efficientnet_b0()

    total, trainable = count_parameters(
        efficientnet_model
    )

    print("\nEfficientNet-B0")
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")


    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL FOUR MODELS LOADED SUCCESSFULLY")
    print("=" * 60)