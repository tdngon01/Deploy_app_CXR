import csv
import json
import os
import random
import time
import psutil
import numpy as np
import torch
from sklearn.metrics import *
from torch import optim
from torch.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modules.config import System_Config as cfg


def set_seed(seed): #cố định số ngẫu nhiên 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def optimizer_to_device(optimizer, device): #đưa qua GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def save_checkpoint(*, save_path, epoch, model, optimizer, scheduler, scaler, best_auc, extra):
    ckpt = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "auc": float(best_auc),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, save_path)

def resume_checkpoint(resume_path, model, optimizer, scheduler, scaler, device):
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"Không tìm thấy checkpoint để resume: {resume_path}")

    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_to_device(optimizer, device)

    if ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"Không thể load scheduler state: {e}")

    if ckpt.get("scaler") is not None:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception as e:
            print(f"Không thể load AMP scaler: {e}")

    start_epoch = int(ckpt.get("epoch", 0))
    best_auc = float(ckpt.get("auc", 0.0))
    print(f"Tiếp tục từ: {resume_path}")
    print(f"Epoch đã hoàn thành: {start_epoch}")
    print(f"Best auc đã lưu: {best_auc:.4f}")
    return start_epoch, best_auc

TEMP_VRAM = cfg.temp_vram
MODEL_TYPE = cfg.TYPE
writer = SummaryWriter(os.path.join(cfg.LOGS_DIR, f"vram_train_{MODEL_TYPE}_{TEMP_VRAM}_{cfg.FINE_TUNE_CONFIG['MODEL_NAME']}"))
def train(model, train_loader, criterion, optimizer, scaler, device, amp_enabled, epoch, total_epochs, train_mode_name):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=False)# chuyển tensor CPU sang GPU
        labels = labels.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)#Reset gradients về 0 trước backward pass
        with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            output = model(images)
            if hasattr(output, 'logits'):
                loss = criterion(output.logits.float(), labels) # Chỉ lấy đầu ra cuối cùng
            else:
                loss = criterion(output.float(), labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        vram_mb = torch.cuda.memory_allocated(device=device) / (1024 ** 2)
        global_step = epoch * len(train_loader) + i
        if train_mode_name == "Full Fine-Tune":
            writer.add_scalar(f'Memory/FULL_VRAM_MB_{cfg.FINE_TUNE_CONFIG["MODEL_NAME"]}', vram_mb, global_step)
        elif train_mode_name == "LoRA Fine-Tune":
            writer.add_scalar(f'Memory/LORA_VRAM_MB_{cfg.FINE_TUNE_CONFIG["MODEL_NAME"]}', vram_mb, global_step)

        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.6f}"})

    avg_loss = total_loss / len(train_loader)
    return avg_loss

def val(model, val_loader, criterion, device, amp_enabled):
    model.eval()
    val_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=False)
            labels = labels.to(device, non_blocking=False)
            with autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                output = model(images)
                if hasattr(output, 'logits'):
                    loss = criterion(output.logits.float(), labels) # Chỉ lấy đầu ra cuối cùng
                else:
                    loss = criterion(output.float(), labels)

            val_loss += loss.item()
            preds.append(torch.sigmoid(output).detach().float().cpu().numpy()) # Xác suất dự đoán cho mỗi lớp
            targets.append(labels.detach().float().cpu().numpy()) # Nhãn thực tế (0 hoặc 1)
            
    avg_val_loss = val_loss / len(val_loader)
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return avg_val_loss, preds, targets

def auc(targets, preds, class_names):
    try:
        class_auc = roc_auc_score(targets, preds, average=None)
        for name, auc in zip(class_names, class_auc):
            print(f"{name}: {auc:.4f}")
        mean_auc = float(class_auc.mean())
    except Exception as e:
        print(f"Lỗi khi tính auc: {e}")
        mean_auc = 0.5

    return mean_auc

def setup_logging(log_dir, tensorboard_dir, metrics_csv_path, resume_path):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    if (resume_path and os.path.exists(metrics_csv_path)):
        csv_mode = "a"
    else:
        csv_mode = "w"
    history = []

    return csv_mode, history

def load_history_resume(csv_mode, metrics_json_path):
    history = []
    if csv_mode == "a" and os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception as e:
            print(f"Không load được metrics json để append: {e}")
            history = []
    return history

def log_epoch(*, writer_tb, csv_writer, csv_file, metrics_json_path,
                history, epoch, avg_loss, avg_val_loss, mean_auc, current_lr, epoch_time, preds, acuracy):
    print(f"\nEpoch {epoch+1} hoàn thành:")
    print(f"Train Loss:{avg_loss:.4f}")
    print(f"Val Loss:{avg_val_loss:.4f}")
    print(f"Val AUC:{mean_auc:.4f}")
    print(f"Time:{epoch_time:.1f}s")
    print(f"LR:{current_lr}")
    print(f"Acuracy: {acuracy:.4f}")
    # TensorBoard
    writer_tb.add_scalar("Loss/Train", avg_loss, epoch + 1)
    writer_tb.add_scalar("Loss/Validation", avg_val_loss, epoch + 1)
    writer_tb.add_scalar("AUC/Validation", mean_auc, epoch + 1)
    writer_tb.add_scalar("Learning_Rate", current_lr, epoch + 1)
    writer_tb.add_scalar("Time/Epoch", epoch_time, epoch + 1)
    writer_tb.add_scalar("Acuracy", acuracy, epoch + 1)
    # CSV
    csv_writer.writerow([epoch + 1, avg_loss, avg_val_loss, float(mean_auc), current_lr, acuracy, epoch_time])
    csv_file.flush()
    # JSON
    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "val_loss": avg_val_loss,
        "val_auc": float(mean_auc),
        "lr": current_lr,
        "acuracy": acuracy,
        "time": epoch_time,
    })
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    # Stats
    print(f"Mean: {preds.mean():.4f}")
    print(f"Std:  {preds.std():.4f}")

def save_run_info(run_info_path, info_dict):
    os.makedirs(os.path.dirname(run_info_path), exist_ok=True)
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=2)
    return info_dict

def run_training(*,model, train_loader, val_loader, optimizer, device, log_dir, tensorboard_dir,
    metrics_csv_path, metrics_json_path, checkpoint_dir, run_info, ckpt_extra_fn, best_ckpt,
    final_ckpt, train_mode_name, resume_path=None):
    
    criterion = BCEWithLogitsLoss().to(device)
    # AMP
    amp_enabled = bool(cfg.USE_AMP) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=2,
    )
    # Chạy tiếp hoặc bắt đầu mới
    start_epoch = 0
    best_auc = 0.0
    if resume_path:
        start_epoch, best_auc = resume_checkpoint(resume_path, model, optimizer, scheduler, scaler, device)

    total_epochs = int(cfg.FINE_TUNE_CONFIG["EPOCHS"])
    if start_epoch >= total_epochs:
        print(f"Checkpoint đã ở epoch {start_epoch}.")
        return

    print(f"\nBắt đầu huấn luyện {train_mode_name}...\n")
    # Logging setup
    csv_mode, _ = setup_logging(log_dir, tensorboard_dir, metrics_csv_path, resume_path)
    history = load_history_resume(csv_mode, metrics_json_path)
    with open(metrics_csv_path, csv_mode, newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        writer_tb = SummaryWriter(tensorboard_dir)
        if csv_mode == "w":
            csv_writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_auc",
                "lr",
                "acuracy",
                "epoch_time_sec",
            ])

        for epoch in range(start_epoch, total_epochs):
            start_time = time.time()
            # Train
            avg_loss = train(
                model, train_loader, criterion, optimizer, scaler,
                device, amp_enabled, epoch, total_epochs, train_mode_name
            )
            # Validation
            avg_val_loss, preds, targets = val(model, val_loader, criterion, device, amp_enabled)
            print(f"Loss train {avg_loss:.4f}  val {avg_val_loss:.4f}:")
            # AUC
            mean_auc = auc(targets, preds, cfg.CLASS_NAMES)
            acuracy = accuracy_score(targets, (preds > 0.5).astype(int))
            epoch_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(mean_auc)
            # Log
            log_epoch(
                writer_tb=writer_tb,
                csv_writer=csv_writer,
                csv_file=csv_file,
                metrics_json_path=metrics_json_path,
                history=history,
                epoch=epoch,
                avg_loss=avg_loss,
                avg_val_loss=avg_val_loss,
                mean_auc=mean_auc,
                current_lr=current_lr,
                acuracy=acuracy,
                epoch_time=epoch_time,
                preds=preds,
            )
            # Checkpoint
            ckpt_extra = ckpt_extra_fn(run_info)
            # Lưu best AUC
            if mean_auc > best_auc:
                best_auc = float(mean_auc)
                save_checkpoint(
                    save_path=os.path.join(checkpoint_dir, f"{epoch+1}_{best_ckpt}.pth.tar"),
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_auc=best_auc,
                    extra=ckpt_extra,
                )
                print(f"AUC cao nhất tại epoch {epoch+1}: {best_auc:.4f}")
        # Lưu checkpoint cuối cùng
        save_name = f"{final_ckpt}_{best_auc:.4f}.pth.tar"
        save_checkpoint(
            save_path=os.path.join(checkpoint_dir, save_name),
            epoch=total_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_auc=best_auc,
            extra=ckpt_extra_fn(run_info),
        )
        print(f"\nAUC cao nhất đạt được: {best_auc:.4f}")
        print(f"Huấn luyện {train_mode_name} hoàn tất")
        writer_tb.close()