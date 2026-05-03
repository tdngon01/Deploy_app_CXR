import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    "Kiến trúc": [
        "EfficientNet-B0", "EfficientNet-B0",
        "ResNet-18", "ResNet-18",
        "MobileNet-V2", "MobileNet-V2",
        "DenseNet-121", "DenseNet-121",
        "GoogleNet", "GoogleNet",
        "VGG16", "VGG16"
    ],
    "Phương pháp": [
        "LoRA", "Full",
        "LoRA", "Full",
        "LoRA", "Full",
        "LoRA", "Full",
        "LoRA", "Full",
        "LoRA", "Full"
    ],
    "AUC tập xác thực": [
        0.9260, 0.9264,
        0.9251, 0.9228,
        0.9212, 0.9139,
        0.9312, 0.9267,
        0.9234, 0.9201,
        0.9021, 0.9353
    ],
    "AUC tập kiểm tra": [
        0.9320, 0.9305,
        0.9262, 0.9268,
        0.9219, 0.9174,
        0.9320, 0.9295,
        0.9282, 0.9235,
        0.9011, 0.9268
    ]
}

df = pd.DataFrame(data)
df["Tên"] = df["Kiến trúc"] + " - " + df["Phương pháp"]

x = np.arange(len(df))
width = 0.40

plt.figure(figsize=(14, 6))

plt.bar(x - width/2, df["AUC tập xác thực"], width, label="AUC tập xác thực")
plt.bar(x + width/2, df["AUC tập kiểm tra"], width, label="AUC tập kiểm tra")

# Hiện số trên đầu cột AUC tập xác thực
for i, v in enumerate(df["AUC tập xác thực"]):
    plt.text(i - width/2, v + 0.0005, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

# Hiện số trên đầu cột AUC tập kiểm tra
for i, v in enumerate(df["AUC tập kiểm tra"]):
    plt.text(i + width/2, v + 0.0005, f"{v:.4f}", ha='center', va='bottom', fontsize=8)

plt.xticks(x, df["Tên"], rotation=45, ha='right')
plt.ylabel("AUC")
plt.title("Biểu đồ cột kết quả AUC tập xác thực và tập kiểm tra")
plt.ylim(0.89, 0.95)
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()