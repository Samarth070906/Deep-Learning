import os
import gc
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

EPOCHS = 3
LEARNING_RATE = 0.001

MODEL_DIR = "saved_models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# MODEL FILE NAMES
# ============================================================

MODEL_FILES = {
    "AlexNet": os.path.join(
        MODEL_DIR,
        "alexnet.pth"
    ),

    "VGG16": os.path.join(
        MODEL_DIR,
        "vgg16.pth"
    ),

    "ResNet50": os.path.join(
        MODEL_DIR,
        "resnet50.pth"
    ),

    "EfficientNet-B0": os.path.join(
        MODEL_DIR,
        "efficientnet_b0.pth"
    )
}


# ============================================================
# CHECK WHETHER ALL MODELS ARE ALREADY SAVED
# ============================================================

all_models_saved = all(
    os.path.exists(path)
    for path in MODEL_FILES.values()
)

results_file = os.path.join(
    RESULT_DIR,
    "transfer_learning_results.csv"
)


# ============================================================
# IF EVERYTHING IS ALREADY DONE
# ============================================================

if all_models_saved and os.path.exists(results_file):

    print("\n" + "=" * 70)
    print("ALL TRAINED MODELS ALREADY EXIST")
    print("=" * 70)

    print("\nNo training is required.")

    print("\nSaved models:")

    for model_name, path in MODEL_FILES.items():
        print(
            f"✓ {model_name}: {path}"
        )

    print("\nSaved results:")

    print(
        f"✓ {results_file}"
    )

    print("\nTo view the results, run:")

    print(
        "python show_results.py"
    )

    print("=" * 70)

    raise SystemExit


# ============================================================
# IMPORT DATA PIPELINE AND MODELS
# ============================================================

from data_pipeline import (
    train_loader,
    val_loader,
    SELECTED_CLASSES,
    BATCH_SIZE,
    device
)

from models import (
    create_alexnet,
    create_vgg16,
    create_resnet50,
    create_efficientnet_b0
)


# ============================================================
# BASIC INFORMATION
# ============================================================

NUM_CLASSES = len(
    SELECTED_CLASSES
)


print("\n" + "=" * 70)
print("TRANSFER LEARNING EXPERIMENT")
print("=" * 70)

print(
    "Device:",
    device
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Training images:",
    len(train_loader.dataset)
)

print(
    "Validation images:",
    len(val_loader.dataset)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    model_name
):

    print("\n" + "=" * 70)
    print(
        f"STARTING TRAINING: {model_name}"
    )
    print("=" * 70)

    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    start_time = time.time()


    # ========================================================
    # EPOCHS
    # ========================================================

    for epoch in range(EPOCHS):

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()

            optimizer.step()


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total += labels.size(0)


        train_loss = (
            running_loss / total
        )

        train_accuracy = (
            correct / total
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        validation_loss = 0.0

        val_correct = 0

        val_total = 0


        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )


                outputs = model(
                    images
                )


                loss = criterion(
                    outputs,
                    labels
                )


                validation_loss += (
                    loss.item()
                    * images.size(0)
                )


                predictions = (
                    outputs.argmax(
                        dim=1
                    )
                )


                val_correct += (
                    predictions == labels
                ).sum().item()


                val_total += labels.size(0)


        val_loss = (
            validation_loss / val_total
        )

        val_accuracy = (
            val_correct / val_total
        )


        # ----------------------------------------------------
        # SAVE HISTORY
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        print(
            f"\nEpoch [{epoch + 1}/{EPOCHS}]"
        )

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy   : "
            f"{val_accuracy * 100:.2f}%"
        )


    training_time = (
        time.time() - start_time
    )


    print(
        f"\n{model_name} training completed."
    )

    print(
        f"Training time: "
        f"{training_time / 60:.2f} minutes"
    )


    return history, training_time


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    model_name
):

    print("\n" + "-" * 70)

    print(
        f"EVALUATING: {model_name}"
    )

    print("-" * 70)


    model.eval()

    all_labels = []

    all_predictions = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )


            outputs = model(
                images
            )


            predictions = (
                outputs
                .argmax(dim=1)
                .cpu()
                .numpy()
            )


            all_predictions.extend(
                predictions
            )

            all_labels.extend(
                labels.numpy()
            )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    precision = precision_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    recall = recall_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )


    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1-Score : {f1 * 100:.2f}%"
    )


    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


# ============================================================
# SAVE MODEL CHECKPOINT
# ============================================================

def save_model(
    model,
    model_name,
    history,
    training_time,
    metrics
):

    checkpoint = {

        "model_name": model_name,

        "model_state_dict":
            model.state_dict(),

        "history": history,

        "training_time":
            training_time,

        "metrics": metrics,

        "selected_classes":
            SELECTED_CLASSES,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE
    }


    torch.save(
        checkpoint,
        MODEL_FILES[model_name]
    )


    print(
        f"\n✓ {model_name} saved to:"
    )

    print(
        MODEL_FILES[model_name]
    )


# ============================================================
# RESULTS LIST
# ============================================================

all_results = []

all_histories = {}


# ============================================================
# FUNCTION TO TRAIN ONE MODEL
# ============================================================

def run_model(
    model_name,
    create_function
):

    # --------------------------------------------------------
    # CHECK WHETHER MODEL ALREADY EXISTS
    # --------------------------------------------------------

    if os.path.exists(
        MODEL_FILES[model_name]
    ):

        print("\n" + "=" * 70)

        print(
            f"{model_name} ALREADY TRAINED"
        )

        print("=" * 70)

        print(
            "Loading saved checkpoint..."
        )


        checkpoint = torch.load(
            MODEL_FILES[model_name],
            map_location=device,
            weights_only=False
        )


        metrics = checkpoint[
            "metrics"
        ]

        history = checkpoint[
            "history"
        ]

        training_time = checkpoint[
            "training_time"
        ]


        all_results.append({

            "Model":
                model_name,

            "Accuracy":
                metrics["Accuracy"],

            "Precision":
                metrics["Precision"],

            "Recall":
                metrics["Recall"],

            "F1-Score":
                metrics["F1-Score"],

            "Training Time (min)":
                training_time / 60
        })


        all_histories[
            model_name
        ] = history


        print(
            "✓ Training skipped."
        )

        print(
            f"Saved Accuracy: "
            f"{metrics['Accuracy'] * 100:.2f}%"
        )


        return


    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\n" + "#" * 70)

    print(
        f"# {model_name}"
    )

    print("#" * 70)


    model = create_function()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    history, training_time = (
        train_model(
            model,
            model_name
        )
    )


    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        model_name
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_model(
        model,
        model_name,
        history,
        training_time,
        metrics
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    all_results.append({

        "Model":
            model_name,

        "Accuracy":
            metrics["Accuracy"],

        "Precision":
            metrics["Precision"],

        "Recall":
            metrics["Recall"],

        "F1-Score":
            metrics["F1-Score"],

        "Training Time (min)":
            training_time / 60
    })


    all_histories[
        model_name
    ] = history


    # --------------------------------------------------------
    # FREE GPU MEMORY
    # --------------------------------------------------------

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# TRAIN ALEXNET
# ============================================================

run_model(
    "AlexNet",
    create_alexnet
)


# ============================================================
# TRAIN VGG16
# ============================================================

run_model(
    "VGG16",
    create_vgg16
)


# ============================================================
# TRAIN RESNET50
# ============================================================

run_model(
    "ResNet50",
    create_resnet50
)


# ============================================================
# TRAIN EFFICIENTNET-B0
# ============================================================

run_model(
    "EfficientNet-B0",
    create_efficientnet_b0
)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    results_file,
    index=False
)


# Save training histories separately.
history_file = os.path.join(
    RESULT_DIR,
    "training_histories.json"
)


with open(
    history_file,
    "w"
) as file:

    json.dump(
        all_histories,
        file,
        indent=4
    )


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n\n")

print("=" * 100)

print(
    "FINAL MODEL COMPARISON"
)

print("=" * 100)


display_df = results_df.copy()


display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
)

display_df["Precision"] = (
    display_df["Precision"] * 100
)

display_df["Recall"] = (
    display_df["Recall"] * 100
)

display_df["F1-Score"] = (
    display_df["F1-Score"] * 100
)


display_df["Accuracy"] = (
    display_df["Accuracy"]
    .map(lambda x: f"{x:.2f}%")
)

display_df["Precision"] = (
    display_df["Precision"]
    .map(lambda x: f"{x:.2f}%")
)

display_df["Recall"] = (
    display_df["Recall"]
    .map(lambda x: f"{x:.2f}%")
)

display_df["F1-Score"] = (
    display_df["F1-Score"]
    .map(lambda x: f"{x:.2f}%")
)

display_df["Training Time (min)"] = (
    display_df["Training Time (min)"]
    .map(lambda x: f"{x:.2f}")
)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# FIND BEST MODEL
# ============================================================

best_model = results_df.iloc[0]


print("\n" + "=" * 70)

print(
    f"BEST MODEL: "
    f"{best_model['Model']}"
)

print(
    f"Accuracy: "
    f"{best_model['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision: "
    f"{best_model['Precision'] * 100:.2f}%"
)

print(
    f"Recall: "
    f"{best_model['Recall'] * 100:.2f}%"
)

print(
    f"F1-Score: "
    f"{best_model['F1-Score'] * 100:.2f}%"
)

print(
    f"Training Time: "
    f"{best_model['Training Time (min)']:.2f} minutes"
)

print("=" * 70)


# ============================================================
# GRAPH 1: ACCURACY
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    results_df["Model"],
    results_df["Accuracy"] * 100
)

plt.title(
    "Validation Accuracy Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "accuracy_comparison.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 2: PRECISION / RECALL / F1
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1-Score"
]

x = range(
    len(results_df)
)

width = 0.25


plt.figure(
    figsize=(12, 6)
)


for i, metric in enumerate(
    metrics
):

    plt.bar(
        [
            value
            + (i - 1) * width
            for value in x
        ],

        results_df[metric] * 100,

        width=width,

        label=metric
    )


plt.title(
    "Precision, Recall and F1-Score Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Score (%)"
)

plt.xticks(
    list(x),
    results_df["Model"],
    rotation=20
)

plt.ylim(
    0,
    100
)

plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "metrics_comparison.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 3: TRAINING TIME
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    results_df["Model"],
    results_df["Training Time (min)"]
)

plt.title(
    "Training Time Comparison"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "Training Time (minutes)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "training_time_comparison.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# GRAPH 4: ACCURACY CURVES
# ============================================================

for model_name, history in all_histories.items():

    epochs = range(
        1,
        len(
            history["train_accuracy"]
        ) + 1
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        epochs,

        [
            value * 100
            for value in
            history["train_accuracy"]
        ],

        marker="o",

        label="Training Accuracy"
    )


    plt.plot(
        epochs,

        [
            value * 100
            for value in
            history["val_accuracy"]
        ],

        marker="o",

        label="Validation Accuracy"
    )


    plt.title(
        f"{model_name} Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.ylim(
        0,
        100
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    filename = (
        model_name
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_accuracy.png"
    )


    plt.savefig(
        os.path.join(
            RESULT_DIR,
            filename
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# GRAPH 5: LOSS CURVES
# ============================================================

for model_name, history in all_histories.items():

    epochs = range(
        1,
        len(
            history["train_loss"]
        ) + 1
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        epochs,

        history["train_loss"],

        marker="o",

        label="Training Loss"
    )


    plt.plot(
        epochs,

        history["val_loss"],

        marker="o",

        label="Validation Loss"
    )


    plt.title(
        f"{model_name} Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.grid(
        True
    )

    plt.tight_layout()


    filename = (
        model_name
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        + "_loss.png"
    )


    plt.savefig(
        os.path.join(
            RESULT_DIR,
            filename
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 90)

print(
    "TRANSFER LEARNING EXPERIMENT COMPLETED"
)

print("=" * 90)

print("\nSaved model files:")

for model_name, path in MODEL_FILES.items():

    print(
        f"✓ {model_name}: {path}"
    )


print("\nSaved result files:")

print(
    f"✓ {results_file}"
)

print(
    f"✓ {history_file}"
)

print(
    "✓ accuracy_comparison.png"
)

print(
    "✓ metrics_comparison.png"
)

print(
    "✓ training_time_comparison.png"
)

print("\nTo view results without retraining:")

print(
    "python show_results.py"
)

print("=" * 90)