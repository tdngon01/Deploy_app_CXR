# https://www.kaggle.com/datasets/nih-chest-xrays/sample
# https://www.kaggle.com/datasets/rishabhrp/chest-x-ray-dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    df = pd.read_csv(r'D:\Khoa_Luan\Du_Lieu\archive\Data_Entry_2017.csv')
    df = df[df['Finding Labels'] == 'No Finding'].copy()

    df = df[['Image Index', 'Finding Labels', 'Patient ID']]
    df = df.rename(columns = {'Image Index': 'image_id'})

    np.random.seed(42)
    patient_ids = df['Patient ID'].unique()
    np.random.shuffle(patient_ids)

    n_pretrain = 10000
    pretrain_ids = patient_ids[:n_pretrain]
    df_pretrain = df[df['Patient ID'].isin(pretrain_ids)]
    df_pretrain.to_csv(r'data/train_csv/data_csv10/pre_train.csv', index=False)
    print(f"Pre-train có: {len(df_pretrain)} ảnh")

if __name__ == "__main__":
    main()