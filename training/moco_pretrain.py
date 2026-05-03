# Nguồn tham khảo:
# https://github.com/facebookresearch/moco/blob/main/main_moco.py
# https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
import os
import time
import torch
import csv
from modules.moco import MoCo
from torch import nn, optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from modules.config import System_Config as cfg
from modules import dataset

device = cfg.DEVICE

def main():
    print(f"Đang sử dụng thiết bị: {device}")
    if not os.path.exists(cfg.CHECKPOINT_SAVE_PRETRAIN):
        os.makedirs(cfg.CHECKPOINT_SAVE_PRETRAIN, exist_ok=True)
        print(f"Đã tạo thư mục lưu model: {cfg.CHECKPOINT_SAVE_PRETRAIN}")
    # tạo log
    LOG_DIR = os.path.join(cfg.LOGS_DIR, f"pretrain_{cfg.PRETRAIN_CONFIG['BACKBONE']}")
    os.makedirs(LOG_DIR, exist_ok=True)
    METRICS_CSV_PATH = os.path.join(LOG_DIR, "metrics_pretrain.csv")

    moco_dataloader, _ = dataset.Pre_Train_DataLoader()
    print("Dữ liệu huấn luyện đã sẵn sàng.")
    print(f"Số lượng batch: {len(moco_dataloader)}")
    
    model_moco = MoCo().to(device)
    print(f"Model MoCo, kiến trúc {cfg.PRETRAIN_CONFIG['BACKBONE']} đã được khởi tạo")

    criterion = nn.CrossEntropyLoss().to(device) #loss
    optimizer = torch.optim.SGD( # hàm tối ưu, hội tụ chậm nhưng ổn định
        model_moco.parameters(),
        lr=cfg.PRETRAIN_CONFIG["LR"],
        momentum=0.9,
        weight_decay=cfg.PRETRAIN_CONFIG["WEIGHT_DECAY"],
    )
    # giảm dần lr
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.PRETRAIN_CONFIG["EPOCHS"],
        eta_min=1e-6,
    )
    scaler = GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")

    start_epoch = 0
    resume_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "last.pth.tar")
    if os.path.exists(resume_path):
        print(f"\nTìm thấy checkpoint cũ: {resume_path}")
        print("Đang khôi phục huấn luyện...")

        checkpoint = torch.load(resume_path, map_location=device)
        model_moco.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        start_epoch = checkpoint['epoch']
        print(f"Thành công. Sẽ bắt đầu từ Epoch {start_epoch + 1}")
    else:
        print("\nKhông tìm thấy checkpoint. Bắt đầu huấn luyện mới.")

    print("\nBắt đầu huấn luyện với MoCo...\n")
    csv_mode = "a" if os.path.exists(METRICS_CSV_PATH) and os.path.exists(resume_path) else "w"
    with open(METRICS_CSV_PATH, csv_mode, newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        if csv_mode == "w":
            csv_writer.writerow(["epoch", "loss", "lr", "epoch_time_sec"])

        for epoch in range(start_epoch, cfg.PRETRAIN_CONFIG["EPOCHS"]):
            model_moco.train()
            total_loss = 0.0
            start_time = time.time()
            pbar = tqdm(
                moco_dataloader,
                desc=f"Epoch {epoch+1}/{cfg.PRETRAIN_CONFIG['EPOCHS']}",
            )
            for i, images in enumerate(pbar):
                img_q = images[0].to(device, non_blocking=True)
                img_k = images[1].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True) # xóa gradient cũ
                with autocast(device_type="cuda" if device.type == "cuda" else "cpu",enabled=cfg.USE_AMP):
                    output, target = model_moco(img_q, img_k)
                    loss = criterion(output, target.long())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "LR":   f"{current_lr:.6f}",
                })
            # tính epoch
            avg_loss = total_loss / len(moco_dataloader)
            duration = time.time() - start_time
            current_lr_epoch = optimizer.param_groups[0]["lr"]
    
            csv_writer.writerow([epoch + 1, avg_loss, current_lr_epoch, duration])
            f.flush()
            # cập nhật lr
            print(f"Epoch {epoch+1} hoàn thành. Loss trung bình: {avg_loss:.4f}. Thời gian: {duration:.1f}s")
            scheduler.step()
            print("LR chuẩn bị cho epoch sau:", optimizer.param_groups[0]["lr"])
            # Lưu checkpoint
            checkpoint_dict = {
                'epoch': epoch + 1,
                'state_dict': model_moco.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'loss': avg_loss,
                'duration': duration,
                'config': cfg.PRETRAIN_CONFIG,
            }
            torch.save(checkpoint_dict, os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, "last.pth.tar"))
            if (epoch + 1) == cfg.PRETRAIN_CONFIG["EPOCHS"]:
                save_path = os.path.join(cfg.CHECKPOINT_SAVE_PRETRAIN, f"pretrain_moco_epoch_{epoch+1}.pth.tar")
                torch.save(checkpoint_dict, save_path)
                print(f"Đã lưu checkpoint định kỳ: {save_path}")

    print("\nHuấn luyện hoàn tất")

if __name__ == "__main__":
    main()

