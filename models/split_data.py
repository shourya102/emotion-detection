import os
import shutil

import pandas as pd

from models.train import train_annotations_path, test_annotations_path, valid_annotations_path

splits = ["train", "test", "val"]
categories = ["body", "context"]
input_dir = '/images'
output_dir = '/images_split'
df_train = pd.read_csv(train_annotations_path)
df_val = pd.read_csv(valid_annotations_path)
df_test = pd.read_csv(test_annotations_path)

for split in splits:
    for category in categories:
        os.makedirs(os.path.join(output_dir, split, category), exist_ok=True)


def split_npy_files(df, image_folder, output_folder):
    for _, row in df.iterrows():
        arr_name = row["Arr_name"]
        crop_name = row["Crop_name"]
        if "train" in crop_name:
            split_type = "train"
        elif "test" in crop_name:
            split_type = "test"
        elif "val" in crop_name:
            split_type = "val"
        else:
            continue
        context_src = os.path.join(image_folder, arr_name)
        context_dest = os.path.join(output_folder, split_type, "context", arr_name)
        if os.path.exists(context_src):
            shutil.move(context_src, context_dest)
        body_src = os.path.join(image_folder, crop_name)
        body_dest = os.path.join(output_folder, split_type, "body", crop_name)
        if os.path.exists(body_src):
            shutil.move(body_src, body_dest)
