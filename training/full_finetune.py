import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torchvision.models as models
from torch import nn, optim
import modules.dataset as dataset
from modules.config import System_Config as cfg
from modules.trainer import *

if cfg.TYPE == "moco":
    from modules.utils import load_moco_pretrained_weights as load_weights
    PRETRAIN_CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_moco_epoch_20.pth.tar")
else:
    from modules.utils import load_spark_pretrained_weights as load_weights
    PRETRAIN_CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "pretrain_spark_epoch_20.pth.tar")

DEVICE = cfg.DEVICE
LOG_DIR = os.path.join(cfg.LOGS_DIR, f"full_finetune_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}")
TENSORBOARD_DIR = os.path.join(LOG_DIR, f"tensorboard_full_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}")
METRICS_CSV_PATH = os.path.join(LOG_DIR, f"metrics_full_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}.csv")
METRICS_JSON_PATH = os.path.join(LOG_DIR, f"metrics_full_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}.json")
RUN_INFO_PATH = os.path.join(LOG_DIR, f"run_info_full_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}.json")

CHECKPOINT_DIR = os.path.join(cfg.CHECKPOINT_DIR, f"full_finetune_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}")
# RESUME_PATH = os.path.join(CHECKPOINT_DIR, "19_full_best_auc.pth.tar")
RESUME_PATH = None
SUPPORTED_BACKBONES = ("EfficientNet", "ResNet", "DenseNet", "MobileNet", "GoogleNet", "VGG16")

def model_full():
    backbone = cfg.FINE_TUNE_CONFIG["MODEL_NAME"]
    if backbone == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "ResNet":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "DenseNet":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None, aux_logits=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, cfg.NUM_CLASSES),
        )
    else:
        raise ValueError(
            f"Không có backbone '{backbone}'. Chỉ có các backbone: {', '.join(SUPPORTED_BACKBONES)}"
        )
    model = load_weights(model, PRETRAIN_CHECKPOINT_PATH)
    model = model.to(DEVICE)
    return model

def main():
    print(f"Đang sử dụng thiết bị: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    set_seed(int(cfg.SEED))

    train_dataloader, val_dataloader, test_dataloader = dataset.Fine_Tune_DataLoader()
    print("Dữ liệu huấn luyện đã sẵn sàng.")
    print(f"Số lượng batch xác thực: {len(val_dataloader)}")
    print(f"Số lượng batch huấn luyện: {len(train_dataloader)}")
    print(f"Số lượng batch kiểm tra: {len(test_dataloader)}")
    print(f"Số bệnh được huấn luyện: {cfg.NUM_CLASSES}")
    # số mẫu dương
    counts = {}
    for disease in cfg.CLASS_NAMES:
        try:
            counts[disease] = dataset.count_samples(cfg.TRAIN_CSV, disease)
        except Exception as e:
            counts[disease] = None
            print(f"Không đếm được số mẫu cho '{disease}': {e}")

    if counts:
        print("Thống kê số mẫu dương tính trong CSV:")
        for k, v in counts.items():
            print(f"- {k}: {v}")

    model = model_full()
    print("Model đã được khởi tạo và tải trọng số tiền huấn luyện")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) #tham số huấn luyện
    print(f"Tổng số tham số phải huấn luyện: {total_params:,} tham số")
    # Lưu
    run_info = save_run_info(RUN_INFO_PATH, {
        "seed": int(cfg.SEED),
        "device": str(DEVICE),
        "num_classes": int(cfg.NUM_CLASSES),
        "class_names": cfg.CLASS_NAMES,
        "img_size": int(cfg.IMG_SIZE),
        "mean": cfg.MEAN,
        "std": cfg.STD,
        "train_csv": cfg.TRAIN_CSV,
        "val_csv": cfg.VAL_CSV,
        "test_csv": cfg.TEST_CSV,
        "image_root": cfg.TRAIN_FINE_TUNE_DIR_IMG,
        "checkpoint_pretrain": PRETRAIN_CHECKPOINT_PATH,
        "fine_tune_config": cfg.FINE_TUNE_CONFIG,
        "counts_train_csv": counts,
        "train_samples": len(train_dataloader.dataset),
        "val_samples": len(val_dataloader.dataset),
        "test_samples": len(test_dataloader.dataset),
        "train_batches": len(train_dataloader),
        "val_batches": len(val_dataloader),
        "test_batches": len(test_dataloader),
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_mode": "full_finetune",
    })
    print(f"Đã lưu run info: {RUN_INFO_PATH}")
    optimizer = optim.AdamW(# hàm tối ưu, thường hội tụ nhanh hơn SGD nhưng có thể kém ổn định hơn
        model.parameters(),
        lr=cfg.FINE_TUNE_CONFIG["LR_FULL"],
        weight_decay=cfg.FINE_TUNE_CONFIG["WEIGHT_DECAY_FULL"],
    )
    # Checkpoint
    def ckpt_extra_fn(run_info):
        return {
            "run_info_path": RUN_INFO_PATH,
            "run_info": run_info,
            "num_classes": int(cfg.NUM_CLASSES),
            "class_names": cfg.CLASS_NAMES,
            "fine_tune_config": cfg.FINE_TUNE_CONFIG,
        }
    
    run_training(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        optimizer=optimizer,
        device=DEVICE,
        log_dir=LOG_DIR,
        tensorboard_dir=TENSORBOARD_DIR,
        metrics_csv_path=METRICS_CSV_PATH,
        metrics_json_path=METRICS_JSON_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
        run_info=run_info,
        ckpt_extra_fn=ckpt_extra_fn,
        best_ckpt="full_best_auc",
        final_ckpt="full_finetune_best_auc",
        train_mode_name="Full Fine-Tune",
        resume_path=RESUME_PATH,
    )

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    main()
