import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models
from modules import dataset
from modules.config import System_Config as cfg
from modules.utils import load_checkpoint_model
from deployment.style import *

#Chạy: streamlit run deployment/app.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED_BACKBONES = ("EfficientNet", "ResNet", "DenseNet", "MobileNet","GoogleNet", "VGG16")

def get_modules_lora(model, backbone):
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

def build_model(backbone, num_classes):
    if backbone == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif backbone == "ResNet":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif backbone == "DenseNet":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif backbone == "MobileNet":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif backbone == "GoogleNet":
        model = models.googlenet(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif backbone == "VGG16":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"Không có backbone '{backbone}'. Chỉ hỗ trợ: {', '.join(SUPPORTED_BACKBONES)}")

    return model

def get_gradcam_layers(model, backbone, is_lora):
    model = model.base_model.model if is_lora else model
    if backbone == "EfficientNet":
        return [model.features[8]]
    if backbone == "ResNet":
        return [model.layer4[-1]]
    if backbone == "DenseNet":
        return [model.features.denseblock4]
    if backbone == "MobileNet":
        return [model.features[18]]
    if backbone == "GoogleNet":
        return [model.inception5b]
    if backbone == "VGG16":
        return [model.features[28]]
    raise ValueError(f"Không hỗ trợ Grad-CAM cho backbone: {backbone}")

def get_threshold_index(index):
    thresholds = 0.5
    if isinstance(thresholds, (list, tuple)):
        return thresholds[index]
    return float(thresholds)

@st.cache_resource
def load_model_lora(ckpt_path, backbone, num_classes):
    model = build_model(backbone, num_classes)

    for param in model.parameters():
        param.requires_grad = False

    if backbone == "GoogleNet" or backbone == "ResNet":
        modules = ["fc"]
    elif backbone == "VGG16":
        modules = ["classifier.6"]
    else:
        modules = ["classifier"]
    lora_config = LoraConfig(
        r=cfg.LORA_CONFIG["RANK"],
        lora_alpha=cfg.LORA_CONFIG["LORA_ALPHA"],
        target_modules=get_modules_lora(model, backbone),
        lora_dropout=cfg.LORA_CONFIG["DROPOUT"],
        bias="none",
        modules_to_save=modules,
    )
    model = get_peft_model(model, lora_config)
    load_checkpoint_model(model, ckpt_path, map_location=DEVICE, strict=False)
    target_layers = get_gradcam_layers(model, backbone, is_lora=True)
    model.to(DEVICE)
    model.eval()
    return model, target_layers

@st.cache_resource
def load_model_full(ckpt_path, backbone, num_classes):
    model = build_model(backbone, num_classes)
    load_checkpoint_model(model, ckpt_path, map_location=DEVICE, strict=False)

    target_layers = get_gradcam_layers(model, backbone, is_lora=False)
    model.to(DEVICE)
    model.eval()
    return model, target_layers

def select_checkpoint():
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    CHECKPOINT_MOCO = os.path.join(BASE_DIR, "checkpoints_moco")

    all_model_options = {
        "1.MoCo_LoRA_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_EfficientNet", "lora_finetune_best_auc_0.9260.pth.tar"),
            "type": "lora",
            "backbone": "EfficientNet",
        },
        "2.MoCo_Full_EfficientNet-B0": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_EfficientNet", "full_finetune_best_auc_0.9264.pth.tar"),
            "type": "full",
            "backbone": "EfficientNet",
        },
        "3.MoCo_LoRA_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_MobileNet", "lora_finetune_best_auc_0.9212.pth.tar"),
            "type": "lora",
            "backbone": "MobileNet",
        },
        "4.MoCo_Full_MobileNet-V2": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_MobileNet", "full_finetune_best_auc_0.9139.pth.tar"),
            "type": "full",
            "backbone": "MobileNet",
        },
        "5.MoCo_LoRA_ResNet-18": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_ResNet", "lora_finetune_best_auc_0.9251.pth.tar"),
            "type": "lora",
            "backbone": "ResNet",
        },
        "6.MoCo_Full_ResNet-18": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_ResNet", "full_finetune_best_auc_0.9228.pth.tar"),
            "type": "full",
            "backbone": "ResNet",
        },
        "7.MoCo_LoRA_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_DenseNet", "lora_finetune_best_auc_0.9312.pth.tar"),
            "type": "lora",
            "backbone": "DenseNet",
        },
        "8.MoCo_Full_DenseNet-121": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_DenseNet", "full_finetune_best_auc_0.9267.pth.tar"),
            "type": "full",
            "backbone": "DenseNet",
        },
        "9.MoCo_LoRA_GoogleNet": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_GoogleNet", "lora_finetune_best_auc_0.9234.pth.tar"),
            "type": "lora",
            "backbone": "GoogleNet",
        },
        "10.MoCo_Full_GoogleNet": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_GoogleNet", "full_finetune_best_auc_0.9201.pth.tar"),
            "type": "full",
            "backbone": "GoogleNet",
        },
        "11.MoCo_LoRA_VGG16": {
            "path": os.path.join(CHECKPOINT_MOCO, "lora_finetune_VGG16", "lora_finetune_best_auc_0.9021.pth.tar"),
            "type": "lora",
            "backbone": "VGG16",
        },
        "12.MoCo_Full_VGG16": {
            "path": os.path.join(CHECKPOINT_MOCO, "full_finetune_VGG16", "full_finetune_best_auc_0.9353.pth.tar"),
            "type": "full",
            "backbone": "VGG16",
        },
    }
    
    if not all_model_options:
        raise FileNotFoundError("Không tìm thấy checkpoint")

    selected_name = st.sidebar.selectbox("Chọn mô hình chẩn đoán", list(all_model_options.keys()))
    return all_model_options[selected_name]

def predict(model, image):
    #xử lý ảnh đầu vào
    transform = dataset.get_transforms(stage="app_demo")
    img_tensor = transform(image)
    # Chuẩn bị định dạng dữ liệu
    img_tensor = img_tensor.unsqueeze(0) #Thêm chiều batch (1, 3, 224, 224)
    img_tensor = img_tensor.to(DEVICE) #Đưa lên GPU (nếu có)
    # Chạy mô hình
    with torch.no_grad():
        logits = model(img_tensor)

    probs = torch.sigmoid(logits) #chuyển đổi logit thành xác suất (0-1)
    probs = probs.squeeze(0) # Bỏ chiều batch dư thừa
    probs = probs.cpu().numpy() #chuyển từ tensor sang numpy

    return probs, img_tensor

def grad_cam(model, target_layers, img_tensor, class_index):
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_index)]
    grayscale_cams = cam(input_tensor=img_tensor, targets=targets)
    return grayscale_cams[0, :]

def tensor_to_rgb(img_tensor):
    image = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

def main():
    st.set_page_config(page_title="Ứng dụng chẩn đoán bệnh lý lồng ngực", layout="wide")
    load_css()
    st.markdown(
        """
        <div class="app-title">Ứng dụng chẩn đoán bệnh lý lồng ngực từ ảnh X-Quang</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Trung tâm điều khiển")
    model_info = select_checkpoint()
    class_names = cfg.CLASS_NAMES
    num_classes = cfg.NUM_CLASSES
    backbone = model_info["backbone"]
    with st.spinner("Đang tải mô hình..."):
        if model_info["type"] == "lora":
            model, target_layers = load_model_lora(model_info["path"], backbone, num_classes,)
        else:
            model, target_layers = load_model_full(model_info["path"], backbone, num_classes)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Danh sách các bệnh:** {num_classes}")
    for name in class_names:
        st.sidebar.markdown(f"{name}")

    file = st.file_uploader("Tải lên ảnh X-quang ngực của bệnh nhân", type=[".PNG", "png"])
    if file is None:
        st.warning("Vui lòng tải lên một ảnh X-quang ngực để chẩn đoán")
        return

    image = Image.open(file).convert("RGB")
    st.image(image, caption="Ảnh X-quang lồng ngực của bệnh nhân", width=300)
    if st.button("Chẩn đoán"):
        prediction, img_tensor = predict(model, image)
        st.success("Chẩn đoán hoàn tất")
        st.write("Kết quả chẩn đoán:")

        for i, name in enumerate(class_names):
            prob = prediction[i]
            thresh = get_threshold_index(i)
            if prob >= thresh:
                is_positive = True
            else:
                is_positive = False
            prediction_card(name, prob, thresh, is_positive)

        st.markdown("---")
        st.subheader("Bản đồ nhiệt")
        img_rgb = tensor_to_rgb(img_tensor)
        column = 5
        for row_start in range(0, len(class_names), column): #cắt 5 ảnh cho 1 hàng
            row_names = class_names[row_start : row_start + column]
            cols = st.columns(len(row_names)) #chia cột
            for col_idx, name in enumerate(row_names):
                class_index = class_names.index(name)
                cam_map = grad_cam(model=model, target_layers=target_layers, img_tensor=img_tensor, class_index=class_index)
                show_image = show_cam_on_image(img_rgb, cam_map, use_rgb=True)
                with cols[col_idx]:
                    st.image(show_image, caption=name, use_container_width=True)

if __name__ == "__main__":
    main()
