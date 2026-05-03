# Nguồn tham khảo:
# https://github.com/stanfordmlgroup/MoCo-CXR/blob/main/moco_pretraining/moco/moco/builder.py
# https://github.com/lukemelas/EfficientNet-PyTorch
# https://docs.pytorch.org/vision/0.22/_modules/torchvision/models/resnet.html
# https://docs.pytorch.org/vision/0.22/_modules/torchvision/models/mobilenetv2.html
# https://docs.pytorch.org/vision/main/models.html

import torch
from torch import nn
from torchvision.models import *
from modules.config import System_Config as cfg

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, mlp, output_dim):
        super(ProjectionHead, self).__init__()
        if mlp:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            )
        else:
            self.projector = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.projector(x)

class EfficientNetBackbone(nn.Module):
    def __init__(self, mlp, pretrained):
        super(EfficientNetBackbone, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()# Bỏ classifier gốc, lấy feature 1280-d
        dim = cfg.PRETRAIN_CONFIG["DIM"]
        self.projector = ProjectionHead(feature_dim, mlp, dim)
    def forward(self, x):
        features = self.backbone(x)# Đặc trưng 1280-d từ backbone
        projections = self.projector(features)# Vector DIM-d cho contrastive loss
        return projections

class ResNetBackbone(nn.Module):
    def __init__(self, mlp, pretrained):
        super(ResNetBackbone, self).__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        dim = cfg.PRETRAIN_CONFIG["DIM"]
        self.projector = ProjectionHead(feature_dim, mlp, dim)
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

class MobileNetBackbone(nn.Module):
    def __init__(self, mlp, pretrained):
        super(MobileNetBackbone, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()# Bỏ classifier gốc, lấy feature 1280-d
        dim = cfg.PRETRAIN_CONFIG["DIM"]
        self.projector = ProjectionHead(feature_dim, mlp, dim)
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

class DenseNetBackbone(nn.Module):
    def __init__(self, mlp, pretrained):
        super(DenseNetBackbone, self).__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = densenet121(weights=weights)
        feature_dim = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        dim = cfg.PRETRAIN_CONFIG["DIM"]
        self.projector = ProjectionHead(feature_dim, mlp, dim)
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections
    
class GoogLeNetBackbone(nn.Module):
    def __init__(self, mlp, pretrained, aux_logits=False):
        super(GoogLeNetBackbone, self).__init__()
        weights = GoogLeNet_Weights.DEFAULT if pretrained else None
        self.backbone = googlenet(weights=weights, aux_logits=aux_logits)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        dim = cfg.PRETRAIN_CONFIG["DIM"]
        self.projector = ProjectionHead(feature_dim, mlp, dim)
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

class VGG16Backbone(nn.Module):
    def __init__(self, mlp, pretrained):
        super(VGG16Backbone, self).__init__()
        weights = VGG16_Weights.DEFAULT if pretrained else None
        self.backbone = vgg16(weights=weights)
        feature_dim = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Identity()# Bỏ classifier gốc, lấy feature 4096-d
        dim = cfg.PRETRAIN_CONFIG["DIM"]
        self.projector = ProjectionHead(feature_dim, mlp, dim)
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projector(features)
        return projections

# class AlexNetBackbone(nn.Module):
#     def __init__(self, mlp, pretrained):
#         super(AlexNetBackbone, self).__init__()
#         weights = AlexNet_Weights.DEFAULT if pretrained else None
#         self.backbone = alexnet(weights=weights)
#         feature_dim = self.backbone.in_features
#         self.backbone.classifier[6] = nn.Identity()# Bỏ classifier gốc, lấy feature 4096-d
#         dim = cfg.PRETRAIN_CONFIG["DIM"]
#         self.projector = ProjectionHead(feature_dim, mlp, dim)
#     def forward(self, x):
#         features = self.backbone(x)
#         projections = self.projector(features)
#         return projections

class MoCo(nn.Module):
    def __init__(self,
                dim = cfg.PRETRAIN_CONFIG["DIM"],
                K = cfg.PRETRAIN_CONFIG["K"],
                m = cfg.PRETRAIN_CONFIG["m"],
                T = cfg.PRETRAIN_CONFIG["T"],
                mlp = cfg.PRETRAIN_CONFIG["mlp"],
                pretrained = cfg.PRETRAIN_CONFIG["pretrained"]):
        super(MoCo, self).__init__()
        self.K = K # Kích thước hàng đợi âm
        self.m = m # Hệ số momentum 
        self.T = T # Nhiệt độ (temperature)
        self.backbone = cfg.PRETRAIN_CONFIG["BACKBONE"]
        if self.backbone == "EfficientNet":
            self.encoder_q = EfficientNetBackbone(pretrained=pretrained, mlp=mlp) # Query encoder - cập nhật bằng gradient
            self.encoder_k = EfficientNetBackbone(pretrained=pretrained, mlp=mlp) # Key encoder - cập nhật bằng momentum
        elif self.backbone == "ResNet":
            self.encoder_q = ResNetBackbone(pretrained=pretrained, mlp=mlp)
            self.encoder_k = ResNetBackbone(pretrained=pretrained, mlp=mlp)
        elif self.backbone == "MobileNet":
            self.encoder_q = MobileNetBackbone(pretrained=pretrained, mlp=mlp)
            self.encoder_k = MobileNetBackbone(pretrained=pretrained, mlp=mlp)
        elif self.backbone == "DenseNet":
            self.encoder_q = DenseNetBackbone(pretrained=pretrained, mlp=mlp)
            self.encoder_k = DenseNetBackbone(pretrained=pretrained, mlp=mlp)
        elif self.backbone == "GoogLeNet":
            self.encoder_q = GoogLeNetBackbone(pretrained=pretrained, mlp=mlp, aux_logits=False)
            self.encoder_k = GoogLeNetBackbone(pretrained=pretrained, mlp=mlp, aux_logits=False)
        elif self.backbone == "VGG16":
            self.encoder_q = VGG16Backbone(pretrained=pretrained, mlp=mlp)
            self.encoder_k = VGG16Backbone(pretrained=pretrained, mlp=mlp)
        else:
            raise ValueError(
                f"'{self.backbone}'. Chỉ các backbone: EfficientNet, ResNet, DenseNet, MobileNet"
            )
        # Copy trọng số query sang key, tắt gradient cho key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        # Hàng đợi lưu các key âm từ các batch trước
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  # Con trỏ vị trí ghi tiếp theo
            
    @torch.no_grad()
    def momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys): #xóa vector đặc trưng cũ và thêm vector đặc trung mới vào hàng đợi
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            return
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        #mã hóa và chuẩn hóa L2
        q = self.encoder_q(im_q)
        q = nn.functional.normalize(q, dim=1)# chuẩn hóa đưa về 1
        # Key: mã hóa bằng momentum encoder (không gradient)
        with torch.no_grad():
            self.momentum_update_key_encoder()
            # Bỏ bước trộn ảnh ở đây
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            # Bỏ bước khôi phục trộn ảnh ở đây
        # Tích vô hướng cặp dương, cặp âm nc x ck->nk
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # cặp dương: tích vô hướng giữa q và k cùng batch, n là kích thước batch, c là kích thước vector đặc trưng. unsqueeze(-1) để thành 1 cột duy nhất.
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])# cặp âm: tích vô hướng giữa q và tất cả key trong hàng đợi
    
        logits = torch.cat([l_pos, l_neg], dim=1) #gộp theo chiều ngang 2 cái này
        logits /= self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)# Nhãn cột vị trí 0 luôn là cặp dương => Như cái nhãn giả
        self.dequeue_and_enqueue(k)
        return logits, labels
