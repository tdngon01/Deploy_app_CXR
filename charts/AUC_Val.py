import os
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.config import System_Config as cfg

mode = "lora"  # "lora" hoac "full"
model_names = ["EfficientNet", "MobileNet", "ResNet", "DenseNet", "GoogleNet", "VGG16"]
labels = ["EfficientNet-B0", "MobileNet-V2", "ResNet-18", "DenseNet-121", "GoogLeNet", "VGG16"]

plt.figure(figsize=(10, 6))

for model_name, label in zip(model_names, labels):
    csv_path = os.path.join(
        cfg.LOGS_DIR,
        f"{mode}_finetune_{model_name}",
        f"metrics_{mode}_{model_name}.csv",
    )

    df = pd.read_csv(csv_path)
    plt.plot(df["epoch"], df["val_auc"], label=label)

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title(f"Validation AUC - {mode.upper()}")
plt.xlim(1, 20)
plt.xticks(range(2,21,2))
plt.ylim(0.8, 1)
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
