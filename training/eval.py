import os
from os import path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import modules.dataset as dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import *
from torch.amp import autocast
from tqdm import tqdm
from modules.config import System_Config as cfg
from modules.utils import load_checkpoint_model
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_BACKBONES = ("EfficientNet", "ResNet", "DenseNet", "MobileNet", "GoogleNet", "VGG16")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHECKPOINT_MOCO = os.path.join(BASE_DIR, "checkpoints_moco")
Model_info = {
        "1.MoCo_LoRA_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_EfficientNet", "lora_finetune_best_auc_0.9260.pth.tar"),
            "type": "lora_moco",
            "backbone": "EfficientNet",
            "save": "moco_lora_efficientnet_result.csv",
        },
        "2.MoCo_Full_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_EfficientNet", "full_finetune_best_auc_0.9264.pth.tar"),
            "type": "full_moco",
            "backbone": "EfficientNet",
            "save": "moco_full_efficientnet_result.csv",
        },
        "3.MoCo_LoRA_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_MobileNet", "lora_finetune_best_auc_0.9212.pth.tar"),
            "type": "lora_moco",
            "backbone": "MobileNet",
            "save": "moco_lora_mobilenet_result.csv",
        },
        "4.MoCo_Full_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_MobileNet", "full_finetune_best_auc_0.9139.pth.tar"),
            "type": "full_moco",
            "backbone": "MobileNet",
            "save": "moco_full_mobilenet_result.csv",
        },
        "5.MoCo_LoRA_ResNet-18": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_ResNet", "lora_finetune_best_auc_0.9251.pth.tar"),
            "type": "lora_moco",
            "backbone": "ResNet",
            "save": "moco_lora_resnet_result.csv",
        },
        "6.MoCo_Full_ResNet-18": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_ResNet", "full_finetune_best_auc_0.9228.pth.tar"),
            "type": "full_moco",
            "backbone": "ResNet",
            "save": "moco_full_resnet_result.csv",
        },
        "7.MoCo_LoRA_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_DenseNet", "lora_finetune_best_auc_0.9312.pth.tar"),
            "type": "lora_moco",
            "backbone": "DenseNet",
            "save": "moco_lora_densenet_result.csv",
        },
        "8.MoCo_Full_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_DenseNet", "full_finetune_best_auc_0.9267.pth.tar"),
            "type": "full_moco",
            "backbone": "DenseNet",
            "save": "moco_full_densenet_result.csv",
        },
        "9.MoCo_LoRA_GoogleNet": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_GoogleNet", "lora_finetune_best_auc_0.9234.pth.tar"),
            "type": "lora_moco",
            "backbone": "GoogleNet",
            "save": "moco_lora_googlenet_result.csv",
        },
        "10.MoCo_Full_GoogleNet": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_GoogleNet", "full_finetune_best_auc_0.9201.pth.tar"),
            "type": "full_moco",
            "backbone": "GoogleNet",
            "save": "moco_full_googlenet_result.csv",
        },
        "11.MoCo_LoRA_VGG16": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_VGG16", "lora_finetune_best_auc_0.9021.pth.tar"),
            "type": "lora_moco",
            "backbone": "VGG16",
            "save": "moco_lora_vgg16_result.csv",
        },
        "12.MoCo_Full_VGG16": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_VGG16", "full_finetune_best_auc_0.9353.pth.tar"),
            "type": "full_moco",
            "backbone": "VGG16",
            "save": "moco_full_vgg16_result.csv",
        },
    }

def get_lora_target_modules(model, backbone):
    target_modules = []
    if backbone == "EfficientNet":
        prefixes = ("features.6", "features.7", "features.8")
    elif backbone == "ResNet":
        prefixes = ("layer3", "layer4") # chạy lại
    elif backbone == "DenseNet":
        prefixes = ("features.denseblock3", "features.denseblock4") # Chạy lại lora
    elif backbone == "MobileNet":
        prefixes = ("features.12", "features.13", "features.14", "features.15", "features.16", "features.17")
    elif backbone == "GoogleNet":
        prefixes = ("inception4a","inception4b","inception4c","inception4d","inception4e","inception5a", "inception5b")
    elif backbone == "VGG16":
        prefixes = ("features.28")
    else:
        raise ValueError(f"Không có backbone '{backbone}'. Chỉ hỗ trợ: {', '.join(SUPPORTED_BACKBONES)}")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.groups == 1 and name.startswith(prefixes):
            target_modules.append(name)
    
    return target_modules

def load_model_full(backbone, checkpoint_path):
    if backbone == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
    elif backbone == "ResNet":
        model = models.resnet18(weights=None)
    elif backbone == "DenseNet":
        model = models.densenet121(weights=None)
    elif backbone == "MobileNet":
        model = models.mobilenet_v2(weights=None)
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
    else:
        raise ValueError(
            f"Không có backbone '{backbone}'. Chỉ có các backbone: {', '.join(SUPPORTED_BACKBONES)}"
        )

    if backbone == "EfficientNet":
        in_features = 1280
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "ResNet":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "DenseNet":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "MobileNet":
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file: {checkpoint_path}")

    load_checkpoint_model(model, checkpoint_path, map_location=device, strict=True)
    model.to(device)
    model.eval()
    return model

def load_model_lora(backbone, checkpoint_path):
    if backbone == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
    elif backbone == "ResNet":
        model = models.resnet18(weights=None)
    elif backbone == "DenseNet":
        model = models.densenet121(weights=None)
    elif backbone == "MobileNet":
        model = models.mobilenet_v2(weights=None)
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
    else:
        raise ValueError(
            f"Không hỗ trợ '{backbone}'. Chỉ: {', '.join(SUPPORTED_BACKBONES)}"
        )

    if backbone == "EfficientNet":
        in_features = 1280
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "ResNet":
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "DenseNet":
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "MobileNet":
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False, init_weights=True)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    else:
        raise ValueError(
            f"Không hỗ trợ '{backbone}'. Chỉ: {', '.join(SUPPORTED_BACKBONES)}"
        )
    
    target_modules = get_lora_target_modules(model, backbone)
    
    if backbone == "ResNet" or backbone == "GoogleNet":
        modules = ["fc"]
    elif backbone == "VGG16":
        modules = ["classifier.6"]
    else:
        modules = ["classifier"]
        
    lora_config = LoraConfig(
        r=cfg.LORA_CONFIG["RANK"],
        lora_alpha=cfg.LORA_CONFIG["LORA_ALPHA"],
        target_modules=target_modules,
        bias="none",
        lora_dropout=cfg.LORA_CONFIG["DROPOUT"],
        modules_to_save=modules,
    )
    model = get_peft_model(model, lora_config)  # gắn LoRA adapter vào model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Không tìm thấy file: {checkpoint_path}")

    load_checkpoint_model(model, checkpoint_path, map_location=device, strict=True)
    model.to(device)
    model.eval()
    return model

def calculate_metrics(y_true, y_pred):
    y_pred_binary = (y_pred >=0.5).astype(int)  # chuyển xác suất -> nhãn 0/1
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred)
    else:
        auc = 0.5

    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)

    return auc, accuracy, f1, recall, precision

def main():
    _, _, test_dataloader = dataset.Fine_Tune_DataLoader()  # chỉ lấy test loader
    print(f"Số lượng mẫu kiểm tra: {len(test_dataloader.dataset)}")

    test_df = pd.read_csv(cfg.TEST_CSV)
    num_total = len(test_df)
    print(f"\nTổng ảnh test: {num_total}")

    for model_name, info in Model_info.items():
        print(f"\n Đang xử lý mô hình: {model_name}")
        backbone = info["backbone"]
        checkpoint_path = info["path"]
        save_path = os.path.join(BASE_DIR, "results_eval", info["save"])
        model_type = info["type"]

        print(f"Đánh giá mô hình: model_type={model_type}, backbone={backbone}")
        print(f"Đang tải checkpoint: {checkpoint_path}")
        VRAM_MOCO = os.path.join(BASE_DIR, "logs_moco")

        writer = SummaryWriter(os.path.join(VRAM_MOCO, f"vram_test_{model_type}_{backbone}"))

        if model_type == "lora_moco":
            model = load_model_lora(backbone, checkpoint_path)
        elif model_type == "full_moco":
            model = load_model_full(backbone, checkpoint_path)
        else:
            raise ValueError(f"model_type không hợp lệ: {model_type}")

        all_probs = []
        all_targets = []
        print("\nBắt đầu chạy trên tập Test...")
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(
                tqdm(test_dataloader, total=len(test_dataloader), desc="Testing", dynamic_ncols=True)
            ):
                images = images.to(device)
                with autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.USE_AMP):
                    outputs = model(images)
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    probs = torch.sigmoid(logits) # [0,1]
                    vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                    writer.add_scalar(f'Memory/TEST_VRAM_MB_{model_name}', vram_mb, batch_idx)

                all_probs.append(probs.float().cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs)#chẩn đoán
        all_targets = np.concatenate(all_targets)#thật

        results = []
        num_classes = all_targets.shape[1]
        for i in range(num_classes):
            class_name = cfg.CLASS_NAMES[i]
            y_true = all_targets[:, i]
            y_pred = all_probs[:, i]
            n_pos = int(y_true.sum())
            n_neg = len(y_true) - n_pos

            auc, accuracy, f1, recall, precision = calculate_metrics(y_true, y_pred)
            results.append({
                "Class": class_name,
                "AUC": auc,
                "Accuracy": accuracy,
                "F1-Score": f1,
                "Recall": recall,
                "Precision": precision,
                "Positives": n_pos,
                "Negatives": n_neg,
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist()
            })
        
        df = pd.DataFrame(results)

        avg_auc_all = df["AUC"].mean()
        avg_acuracy_all = df["Accuracy"].mean()
        avg_f1_all = df["F1-Score"].mean()
        avg_recall_all = df["Recall"].mean()
        avg_precision_all = df["Precision"].mean()

        df.loc[len(df)] = {
            "Class": "Trung bình tất cả", 
            "AUC": avg_auc_all, 
            "Accuracy": avg_acuracy_all, 
            "F1-Score": avg_f1_all, 
            "Recall": avg_recall_all, 
            "Precision": avg_precision_all,
            "Positives": "",
            "Negatives": "",
            "y_true": "",
            "y_pred": ""
        }

        # Lưu
        df.to_csv(save_path, index=False)
        print(f"Kết quả đánh giá đã được lưu tại: {save_path}")
        writer.close()

if __name__ == "__main__":
    main()
