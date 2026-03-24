import os
import numpy as np
from extract import extract_features

DATASET_FOLDER = "dataset"

X = []
y = []

label_map = {}
label_counter = 0

print("🚀 Building dataset...\n")

for website in os.listdir(DATASET_FOLDER):
    website_path = os.path.join(DATASET_FOLDER, website)

    if not os.path.isdir(website_path):
        continue

    # Assign label
    if website not in label_map:
        label_map[website] = label_counter
        label_counter += 1

    label = label_map[website]

    print(f"[INFO] Processing website: {website} → Label {label}")

    for file in os.listdir(website_path):
        if file.endswith(".pcap") or file.endswith(".pcapng"):

            file_path = os.path.join(website_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"❌ Skipping {file}: {e}")

# Convert to numpy
X = np.array(X)
y = np.array(y)

# Save
np.save("X.npy", X)
np.save("y.npy", y)

print("\n✅ Dataset Created Successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Label Map:", label_map)