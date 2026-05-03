# https://www.kaggle.com/datasets/nih-chest-xrays/sample
# https://www.kaggle.com/datasets/rishabhrp/chest-x-ray-dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv(r'D:\Khoa_Luan\Du_Lieu\VinBigData_CXR_01\train_csv\train.csv')
    TARGETS = [
        "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly",
        "Consolidation", "ILD", "Infiltration", "Lung Opacity",
        "Nodule/Mass", "Other lesion", "Pleural effusion",
        "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis",
        "No finding"
    ]
    for i in TARGETS:
        df[i] = (df['class_name'] == i).astype(int) #tạo cột cho bệnh

    df = df.groupby('image_id')[TARGETS].max().reset_index() #hợp nhất nhãn
    print(f"Tổng số ảnh: {len(df)}")

    np.random.seed(42)
    image_id = df['image_id'].values
    train_id, temp_id = train_test_split(
        image_id,
        test_size=0.3,
        random_state=42
    )
    val_id, test_id = train_test_split(
        temp_id,
        test_size=0.5,
        random_state=42
    )
    
    df_train = df[df['image_id'].isin(train_id)] #lấy ảnh thuộc id
    df_val = df[df['image_id'].isin(val_id)]
    df_test = df[df['image_id'].isin(test_id)]

    save_dir = os.path.join(r'data/train_csv')
    df_train.to_csv(os.path.join(save_dir, 'train_split.csv'), index=False)
    df_val.to_csv(os.path.join(save_dir, 'val_split.csv'), index=False)
    df_test.to_csv(os.path.join(save_dir, 'test_split.csv'), index=False)
    print(f"Train: {len(df_train)} ảnh")
    print(f"val: {len(df_val)} ảnh")
    print(f"test: {len(df_test)} ảnh")
    print(f"Đã lưu tại: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    main()