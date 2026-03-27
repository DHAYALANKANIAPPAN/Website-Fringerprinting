import os
import numpy as np
from extract import extract_features

DATASET_PATH = "dataset"

X = []
y = []
label_map = {}

label = 0

print("🚀 Building dataset...")

for website in os.listdir(DATASET_PATH):
    website_path = os.path.join(DATASET_PATH, website)

    if not os.path.isdir(website_path):
        continue

    print(f"[INFO] Processing {website} → Label {label}")

    label_map[website] = label

    for file in os.listdir(website_path):
        if file.endswith(".pcap") or file.endswith(".pcapng"):
            filepath = os.path.join(website_path, file)

            try:
                features = extract_features(filepath)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"[ERROR] {file}: {e}")

    label += 1

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("\n✅ Dataset Created Successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label Map:", label_map)
