from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Change these paths if your files are in a different folder
# UNET_CSV = Path("UNET_260421/UNET_260421_loss.csv")
# QLOSS_TXT = Path("QUNET_260604/qloss_260604.txt")

# SVG_OUTPUT = Path("loss_curves_vector.svg")
# PDF_OUTPUT = Path("loss_curves_vector.pdf")
# EPS_OUTPUT = Path("loss_curves_vector.eps")

if len(sys.argv) < 4:
    print("Not enough arguments")
    exit()

# Usage: loss_graph.py <Classical loss txt file> <Quantum loss txt file> <Output directory (results/<models to be compared>)
LOSS_TXT = Path(sys.argv[1])
QLOSS_TXT = Path(sys.argv[2])

OUTPUT_DIR = Path(sys.argv[3])
SVG_OUTPUT = OUTPUT_DIR / Path("loss_curves_vector.svg")
PDF_OUTPUT = OUTPUT_DIR / Path("loss_curves_vector.pdf")
EPS_OUTPUT = OUTPUT_DIR / Path("loss_curves_vector.eps")

def read_unet_csv(path: Path):
    """
    Reads a CSV file containing one loss value per row.
    Returns epoch numbers and loss values.
    """
    df = pd.read_csv(path, header=None)
    losses = df.iloc[:, 0].astype(float).tolist()
    epochs = list(range(1, len(losses) + 1))
    return epochs, losses


def read_epoch_loss_txt(path: Path):
    """
    Reads text lines like:
    Epoch 1: average loss 0.11779
    Epoch 2: average loss 0.04748

    Returns parsed epoch numbers and loss values.
    """
    epochs = []
    losses = []

    pattern = re.compile(
        r"Epoch\s+(\d+)\s*:\s*average\s+loss\s+([0-9]*\.?[0-9]+)",
        re.IGNORECASE
    )

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    return epochs, losses


def main():
    unet_epochs, unet_losses = read_epoch_loss_txt(LOSS_TXT)
    q_epochs, q_losses = read_epoch_loss_txt(QLOSS_TXT)

    # unet_epochs, unet_losses = unet_epochs[0:25], unet_losses[0:25]
    # q_epochs, q_losses = q_epochs[0:25], q_losses[0:25]

    plt.figure(figsize=(8, 5))

    plt.plot(unet_epochs, unet_losses, linewidth=1.8, label="U-Net loss")

    if q_epochs and q_losses:
        plt.plot(q_epochs, q_losses, linewidth=1.8, label="Q-UNet loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")

    # X-axis uses the 49 CSV epochs, with max shown as 50
    plt.xlim(0, 50)
    plt.xticks(range(1, 51, 5))

    plt.legend()
    plt.tight_layout()

    # SVG and PDF are vector graphics formats
    plt.savefig(SVG_OUTPUT, format="svg", bbox_inches="tight")
    plt.savefig(PDF_OUTPUT, format="pdf", bbox_inches="tight")
    plt.savefig(EPS_OUTPUT, format="eps", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()