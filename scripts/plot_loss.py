import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_version_dir(logs_dir="lightning_logs"):
    if not os.path.exists(logs_dir):
        return None
    versions = []
    for d in os.listdir(logs_dir):
        if d.startswith("version_"):
            try:
                v = int(d.split("_")[1])
                versions.append((v, os.path.join(logs_dir, d)))
            except ValueError:
                pass
    if not versions:
        return None
    # Return path of the highest version number
    return max(versions, key=lambda x: x[0])[1]

def main():
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves from PyTorch Lightning CSV logs")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the metrics.csv file. If not provided, uses the latest version in lightning_logs/")
    parser.add_argument("--output_path", type=str, default="checkpoints/loss_curve.png", help="Path to save the generated plot")
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path:
        latest_dir = get_latest_version_dir()
        if latest_dir:
            csv_path = os.path.join(latest_dir, "metrics.csv")
            print(f"Auto-detected latest log directory: {latest_dir}")
        else:
            print("Error: No log directory found in lightning_logs/ and --csv_path was not specified.")
            sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"Error: Log file not found at {csv_path}")
        sys.exit(1)

    print(f"Loading metrics from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean and aggregate epoch-level metrics
    # Since lightning logs train metrics and val metrics on different steps/rows within the same epoch,
    # we group by epoch and take the first non-null value (or mean) for each metric.
    epoch_df = df.groupby("epoch").agg({
        "train_loss_epoch": "mean",
        "val_loss": "mean",
        "val_acc": "mean"
    }).reset_index()

    # Drop epochs where we don't have both train_loss and val_loss
    epoch_df = epoch_df.dropna(subset=["train_loss_epoch", "val_loss"])

    if epoch_df.empty:
        print("Warning: No epoch-level train and validation loss metrics found. Trying to plot step-level metrics.")
        # Fallback to step-level if epoch-level isn't fully logged yet
        # (e.g. if the training run was interrupted early in the first epoch)
        plt.figure(figsize=(10, 5))
        if "train_loss_step" in df.columns:
            steps_train = df.dropna(subset=["train_loss_step"])
            plt.plot(steps_train["step"], steps_train["train_loss_step"], label="Train Loss (Step)", alpha=0.6)
        if "val_loss" in df.columns:
            steps_val = df.dropna(subset=["val_loss"])
            plt.plot(steps_val["step"], steps_val["val_loss"], label="Val Loss", marker="o", color="red")
        plt.title("Training Loss Curve (Step-level)")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
    else:
        # Create dual-y axis plot or side-by-side plots for Loss and Accuracy
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        line1, = ax1.plot(epoch_df["epoch"], epoch_df["train_loss_epoch"], label="Train Loss", color=color, linewidth=2, marker='s', markersize=4)
        line2, = ax1.plot(epoch_df["epoch"], epoch_df["val_loss"], label="Val Loss", color="tab:orange", linewidth=2, marker='o', markersize=4)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.5)

        lines = [line1, line2]
        labels = [str(line.get_label()) for line in lines]

        # If val_acc is available, plot it on a second y-axis
        if "val_acc" in epoch_df.columns and not epoch_df["val_acc"].isna().all():
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Validation Accuracy', color=color)
            line3, = ax2.plot(epoch_df["epoch"], epoch_df["val_acc"], label="Val Acc", color=color, linewidth=2, linestyle="--", marker='^', markersize=4)
            ax2.tick_params(axis='y', labelcolor=color)
            lines.append(line3)
            labels.append(str(line3.get_label()))

        plt.title("Neural CAT Training Metrics & Loss Curves", fontsize=14, fontweight="bold", pad=15)
        ax1.legend(lines, labels, loc="upper right")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(args.output_path, dpi=300, bbox_inches="tight")
    print(f"Successfully saved loss curve plot to: {args.output_path}")

if __name__ == "__main__":
    main()
