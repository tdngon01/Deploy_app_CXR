import torch
import os
from collections.abc import Mapping
from modules.config import System_Config as cfg

device = cfg.DEVICE

def load_moco_pretrained_weights(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {path}")
    
    print(f"Đang tải trọng số từ checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        # Chỉ lấy encoder_q.backbone, bỏ encoder_k và projector
        if k.startswith("encoder_q.backbone."):
            name = k.replace("encoder_q.backbone.", "")# "encoder_q.backbone.features.0.0.weight" -> "features.0.0.weight"
            new_state_dict[name] = v
        elif k.startswith("encoder_q.") and not k.startswith("encoder_q.backbone."): # Bỏ qua lớp MLP
            continue

    if len(new_state_dict) == 0:
        print("Không trích xuất được weight nào từ checkpoint")
        print(f"Các key mẫu trong checkpoint có dạng: {list(state_dict.keys())[:5]}")
    else:
        print(f"Đã trích xuất {len(new_state_dict)} tensors từ encoder_q.backbone")
        model.load_state_dict(new_state_dict, strict=False)

    print("Tải trọng số hoàn tất")
    return model

def load_spark_pretrained_weights(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {path}")
    
    print(f"Đang tải trọng số từ checkpoint SparK: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        
        # Trích xuất trọng số chuẩn xác của Backbone
        if k.startswith("encoder.backbone."):
            name = k.replace("encoder.backbone.", "")
            new_state_dict[name] = v

    if len(new_state_dict) == 0:
        print("Không trích xuất được weight nào từ checkpoint SparK")
    else:
        print(f"Đã trích xuất {len(new_state_dict)} tensors từ encoder.backbone")
        model.load_state_dict(new_state_dict, strict=False)

    print("Tải trọng số SparK hoàn tất")
    return model

def load_checkpoint_model(model, checkpoint_path, map_location, strict):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = checkpoint  #checkpoint chính là state_dict
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                state_dict = checkpoint[key]
                break
    # checkpoint lưu dạng này module.features.0.weight nhưng model cần dạng này features.0.weight
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        clean_state_dict[new_key] = value
    # Nạp trọng số vào model
    load_result = model.load_state_dict(clean_state_dict, strict=strict)
    return checkpoint, load_result
