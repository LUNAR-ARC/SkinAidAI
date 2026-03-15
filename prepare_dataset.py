import os
import pandas as pd
import shutil
from tqdm import tqdm

DATASET_PATH = "dataset"
CSV_FILE = os.path.join(DATASET_PATH, "HAM10000_metadata.csv")
IMAGE_FOLDER_1 = os.path.join(DATASET_PATH, "HAM10000_images_part_1")
IMAGE_FOLDER_2 = os.path.join(DATASET_PATH, "HAM10000_images_part_2")

OUTPUT_FOLDER = os.path.join(DATASET_PATH, "images")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

metadata = pd.read_csv(CSV_FILE)

for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):

    image_id = row["image_id"] + ".jpg"
    label = row["dx"]

    label_folder = os.path.join(OUTPUT_FOLDER, label)

    os.makedirs(label_folder, exist_ok=True)

    src = os.path.join(IMAGE_FOLDER_1, image_id)

    if not os.path.exists(src):
        src = os.path.join(IMAGE_FOLDER_2, image_id)

    dst = os.path.join(label_folder, image_id)

    shutil.copy(src, dst)

print("Dataset preparation complete")