import os
from modules.config import System_Config as cfg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Các hàng là lớp bệnh, các cột là tên mô hình
model_name = ['EfficientNet-B0', 'ResNet-18', 'DenseNet-121', 'MobileNet-V2', 'GoogLeNet', 'VGG16']
mode = 'full'
if mode == 'lora':
   model_name_file = ['lora_efficientnet', 'lora_resnet', 'lora_densenet', 'lora_mobilenet', 'lora_googlenet', 'lora_vgg16']
   name_table= "Bản đồ nhiệt chỉ số Accuracy theo từng lớp bệnh (LoRA)"
else:
   model_name_file= ['full_efficientnet', 'full_resnet', 'full_densenet', 'full_mobilenet', 'full_googlenet', 'full_vgg16']
   name_table= "Bản đồ nhiệt chỉ số Accuracy theo từng lớp bệnh (Full)"

data = {}
classes = cfg.CLASS_NAMES

for display_name, file_name in zip(model_name, model_name_file):
    # Sửa đường dẫn: thêm 'results_eval' và dùng os.path.join cho an toàn
    file_path = os.path.join(cfg.BASE_DIR, "results_eval", f"moco_{file_name}_result.csv")

    if not os.path.exists(file_path):
        print(f"Không tìm thấy file: {file_path}")
        continue

    df_csv = pd.read_csv(file_path)

    # Lấy AUC cho các lớp bệnh có trong cfg.CLASS_NAMES
    auc = df_csv[df_csv["Class"].isin(classes)]["Accuracy"].tolist()
   
    if len(auc) == len(classes):
        data[display_name] = auc
    else:
        print(f"Cảnh báo: Số lượng Accuracy ({len(auc)}) không khớp với số lượng lớp ({len(classes)}) cho {display_name}")


df = pd.DataFrame(data, index=classes)
   
# 2. Vẽ Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, fmt=".4f", cmap='YlGnBu', linewidths=.5)

plt.title(name_table)
plt.ylabel('Lớp bệnh')
plt.xlabel('Kiến trúc mô hình')
plt.show()
