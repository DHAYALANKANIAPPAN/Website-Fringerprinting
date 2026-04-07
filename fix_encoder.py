"""
Rebuild label_encoder.pkl with all 10 correct class names.
The model was trained on these 10 classes in this exact alphabetical order.

Run:
    python fix_encoder.py
"""
import os, json, pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = "model_artifacts"

# These are your 10 classes in the exact alphabetical order
# that LabelEncoder assigned during training.
# (LabelEncoder always sorts alphabetically)
CORRECT_CLASSES = [
    'BBC News',   # class 0
    'Coursera',   # class 1
    'Discord',    # class 2
    'Disney',     # class 3
    'Dropbox',    # class 4
    'Facebook',   # class 5
    'github',     # class 6
    'pinterest',  # class 7
    'quora',      # class 8
    'tumble',     # class 9
]

# Verify model output matches — load model and check output layer size
print("Verifying model output size...")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

model_path = os.path.join(MODEL_DIR, "best_model.keras")
model = tf.keras.models.load_model(model_path)

# Find the category output layer size
output_shapes = []
if isinstance(model.output, list):
    for out in model.output:
        output_shapes.append(out.shape[-1])
else:
    output_shapes.append(model.output.shape[-1])

print(f"  Model output sizes: {output_shapes}")
n_model_classes = output_shapes[0]

if n_model_classes != len(CORRECT_CLASSES):
    print(f"\n  [MISMATCH] Model has {n_model_classes} output classes but we have {len(CORRECT_CLASSES)} names.")
    print(f"  Adjusting class list to match model size ({n_model_classes})...")

    # If model has different count, trim or pad
    if n_model_classes < len(CORRECT_CLASSES):
        CORRECT_CLASSES = CORRECT_CLASSES[:n_model_classes]
    else:
        # Pad with generic names if somehow more classes
        while len(CORRECT_CLASSES) < n_model_classes:
            CORRECT_CLASSES.append(f"class_{len(CORRECT_CLASSES)}")

print(f"\n  Final class mapping:")
for i, c in enumerate(CORRECT_CLASSES):
    print(f"    {i} → {c}")

# Rebuild the LabelEncoder
enc = LabelEncoder()
enc.fit(CORRECT_CLASSES)

# Verify
assert list(enc.classes_) == CORRECT_CLASSES, "Encoding mismatch!"
assert enc.transform(['BBC News'])[0] == 0
assert enc.transform(['tumble'])[0] == len(CORRECT_CLASSES) - 1

# Save
enc_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
with open(enc_path, "wb") as f:
    pickle.dump(enc, f)
print(f"\n✓ Saved fixed label_encoder.pkl → {enc_path}")

# Also update metadata.json
meta_path = os.path.join(MODEL_DIR, "metadata.json")
with open(meta_path) as f:
    meta = json.load(f)

meta["class_names"] = CORRECT_CLASSES
meta["n_classes"]   = len(CORRECT_CLASSES)

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"✓ Updated metadata.json with {len(CORRECT_CLASSES)} class names")

print("\n" + "="*50)
print("  Done! Now restart app.py")
print("  python app.py")
print("="*50)