import _quiet
import os
import cv2
from feature_extraction_double import extract_double_hand_features

folder = "data/dataset_double/Shanka"
failed = []
total = 0

for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    img = cv2.imread(path)
    if img is None:
        continue
    total += 1
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feats = extract_double_hand_features(img_rgb, static=True)
    if feats is None:
        failed.append(fname)

print(f"{len(failed)} / {total} failed feature extraction")
print("Failed files:", failed[:20])