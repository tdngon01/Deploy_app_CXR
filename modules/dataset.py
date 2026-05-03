# Nguồn tham khảo: 
# https://github.com/facebookresearch/moco/blob/main/moco/loader.py
# https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from modules.config import System_Config as cfg
import pandas as pd

def count_samples(csv_path, disease_name):
    df = pd.read_csv(csv_path)
    if disease_name not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{disease_name}' trong CSV: {csv_path}")

    return int(df[disease_name].sum())

class TwoCropsTransform:
    def __init__(self, base_transform) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x) # Ảnh query
        k = self.base_transform(x) # Ảnh key
        return [q, k]

def get_transforms(stage):
    img_size = cfg.IMG_SIZE
    if stage == 'pre_train_moco':
        augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)), # cắt ngẫu nhiên 20 - 100% ảnh gốc
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)), # xoay ngãu nhiên +-10 độ, dịch chuyển 5% chiều rộng,chiều cao
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)) # độ mờ
            ], p=0.5), #50% làm mờ
            transforms.RandomHorizontalFlip(p=0.5),# Lat ngang xac suat 50%
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
        ])
        return TwoCropsTransform(augmentation)
    elif stage == 'pre_train_spark':
        augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size, scale=(0.67, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
        ])
        return augmentation
    elif stage == 'train':
        augmentation = transforms.Compose([
            # Dua tat ca anh ve cung kich thuoc dau vao.
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10), # xoay ngẫu nhiên
            transforms.ToTensor(), # chuyển HWC (Height, Width, Channel) sang CHW (Channel, Height, Width)
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
        ])
        return augmentation
    elif stage == 'val' or stage == 'test' or stage == 'app_demo':
        augmentation = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),# Chuyen sang tensor
            transforms.Normalize(mean=cfg.MEAN, std=cfg.STD)
        ])
        return augmentation

def build_image_index(root_dir):
    index = {}
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".png"):
                key = os.path.splitext(f)[0]
                index[key] = os.path.join(root, f)
    print(f"Có {len(index)} ảnh trong {root_dir}")
    return index


class Unlabeled_Dataset(Dataset):
    def __init__(self, csv_file, image_index, transform=None, max_samples=None, seed=42):
        self.df = pd.read_csv(csv_file)
        if max_samples is not None and len(self.df) > max_samples:
            self.df = self.df.sample(n=max_samples, random_state=seed).reset_index(drop=True)  

        self.image_ids = self.df["image_id"].str.replace(".png", "", regex=False).tolist()
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_index.get(image_id)
        if img_path is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_id}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image

class Labeled_Dataset(Dataset):
    def __init__(self, csv_file, image_index, class_name, transform=None, seed=42):
        self.df = pd.read_csv(csv_file)
        self.image_ids = self.df["image_id"].str.replace(".png", "", regex=False).tolist()
        self.labels = self.df[class_name].values.astype("float32")
        self.image_index = image_index
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_index.get(image_id)

        if img_path is None:
            raise FileNotFoundError(f"Không thấy ảnh: {image_id}")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label

IMAGE_INDEX_NIH = None
IMAGE_INDEX_VIN = None

def get_image_index_NIH():
    global IMAGE_INDEX_NIH
    if IMAGE_INDEX_NIH is None:
        IMAGE_INDEX_NIH = build_image_index(cfg.TRAIN_PRE_TRAIN_DIR_IMG)
    return IMAGE_INDEX_NIH

def get_image_index_VIN():
    global IMAGE_INDEX_VIN
    if IMAGE_INDEX_VIN is None:
        IMAGE_INDEX_VIN = build_image_index(cfg.TRAIN_FINE_TUNE_DIR_IMG)
    return IMAGE_INDEX_VIN

def Pre_Train_DataLoader():
    moco_dataset = Unlabeled_Dataset(
        csv_file=cfg.PRE_TRAIN_CSV,
        image_index=get_image_index_NIH(),
        transform=get_transforms(stage='pre_train_moco'),
        max_samples=cfg.PRETRAIN_CONFIG["PRE_TRAIN"]
    )
    spark_dataset = Unlabeled_Dataset(
        csv_file=cfg.PRE_TRAIN_CSV,
        image_index=get_image_index_NIH(),
        transform=get_transforms(stage='pre_train_spark'),
        max_samples=cfg.PRETRAIN_CONFIG["PRE_TRAIN"]
    )
    moco_dataloader = torch.utils.data.DataLoader(
        moco_dataset,
        batch_size=cfg.PRETRAIN_CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)
    )
    spark_dataloader = torch.utils.data.DataLoader(
        spark_dataset,
        batch_size=cfg.PRETRAIN_CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)
    )
    return moco_dataloader, spark_dataloader

def Fine_Tune_DataLoader():
    image_index = get_image_index_VIN()
    train_dataset = Labeled_Dataset(
        csv_file=cfg.TRAIN_CSV,
        image_index=image_index,
        class_name=cfg.CLASS_NAMES,
        transform=get_transforms(stage='train'),
    )
    val_dataset = Labeled_Dataset(
        csv_file=cfg.VAL_CSV,
        image_index=image_index,
        class_name=cfg.CLASS_NAMES,
        transform=get_transforms(stage='val'),
    )
    test_dataset = Labeled_Dataset(
        csv_file=cfg.TEST_CSV,
        image_index=image_index,
        class_name=cfg.CLASS_NAMES,
        transform=get_transforms(stage='test'), 
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.FINE_TUNE_CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
        generator=torch.Generator().manual_seed(42)
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.FINE_TUNE_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.FINE_TUNE_CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
