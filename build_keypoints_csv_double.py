import os
import cv2
import pandas as pd
from feature_extraction_double import extract_double_hand_features, FEATURE_DIM
from tqdm import tqdm
import yaml


with open("mudras_double.yaml") as f:
    LABELS = yaml.safe_load(f)["labels"]
DATASET_DIR = "data/dataset_double"
OUTPUT_CSV = "keypoints_double.csv"

rows = []

for label in LABELS:
    cls_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(cls_path):
        print(f"WARNING: no folder for {label}, skipping")
        continue

    valid, failed = 0, 0
    for img_name in tqdm(os.listdir(cls_path), desc=label, leave=False):
        img_path = os.path.join(cls_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            failed += 1
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        feats = extract_double_hand_features(image_rgb, static=True)
        if feats is None or feats.shape[0] != FEATURE_DIM:
            failed += 1
            continue
        row = {f"f{i}": v for i, v in enumerate(feats)}
        row["label"] = label
        rows.append(row)
        valid += 1

    print(f"{label}: {valid} valid, {failed} failed")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows, {df['label'].nunique()} classes -> {OUTPUT_CSV}")