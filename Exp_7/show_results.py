import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RESULT_DIR = "results"

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "transfer_learning_results.csv"
)


# ============================================================
# CHECK RESULTS
# ============================================================

if not os.path.exists(RESULT_FILE):

    print("\n" + "=" * 60)

    print(
        "RESULT FILE NOT FOUND"
    )

    print("=" * 60)

    print(
        "\nPlease run:"
    )

    print(
        "python train.py"
    )

    print(
        "\nfirst to train the models."
    )

    raise SystemExit


# ============================================================
# LOAD RESULTS
# ============================================================

results = pd.read_csv(
    RESULT_FILE
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 90)

print(
    "TRANSFER LEARNING RESULTS"
)

print("=" * 90)


display_df = results.copy()


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


print()

print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results.iloc[0]


print("\n" + "=" * 70)

print(
    "BEST MODEL"
)

print("=" * 70)

print(
    f"Model     : "
    f"{best_model['Model']}"
)

print(
    f"Accuracy  : "
    f"{best_model['Accuracy'] * 100:.2f}%"
)

print(
    f"Precision : "
    f"{best_model['Precision'] * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{best_model['Recall'] * 100:.2f}%"
)

print(
    f"F1-Score  : "
    f"{best_model['F1-Score'] * 100:.2f}%"
)

print(
    f"Time      : "
    f"{best_model['Training Time (min)']:.2f} minutes"
)

print("=" * 70)


# ============================================================
# SHOW SAVED GRAPHS
# ============================================================

graph_files = [
    "accuracy_comparison.png",
    "metrics_comparison.png",
    "training_time_comparison.png"
]


for graph_file in graph_files:

    path = os.path.join(
        RESULT_DIR,
        graph_file
    )

    if os.path.exists(path):

        image = plt.imread(path)

        plt.figure(
            figsize=(10, 6)
        )

        plt.imshow(
            image
        )

        plt.axis(
            "off"
        )

        plt.title(
            graph_file
        )

        plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)

print(
    "RESULTS LOADED SUCCESSFULLY"
)

print("=" * 70)

print(
    "\nNo model training was performed."
)

print(
    "All results were loaded from saved files."
)

print("=" * 70)