"""
Run this script from your project folder to see what's inside your model artifacts.
    python check_artifacts.py
"""
import os, json, pickle
import numpy as np

MODEL_DIR = "model_artifacts"

print("="*55)
print("  Artifact Inspector")
print("="*55)

# ── Check every file ──────────────────────────────────────────────────────────
for fname in os.listdir(MODEL_DIR):
    fpath = os.path.join(MODEL_DIR, fname)
    print(f"\n  FILE: {fname}  ({os.path.getsize(fpath)} bytes)")

    if fname.endswith(".json"):
        with open(fpath) as f:
            data = json.load(f)
        print(f"    Contents: {json.dumps(data, indent=6)}")

    elif fname.endswith(".pkl"):
        with open(fpath, "rb") as f:
            obj = pickle.load(f)
        print(f"    Type   : {type(obj)}")
        if hasattr(obj, "classes_"):
            print(f"    classes_: {list(obj.classes_)}")
        elif hasattr(obj, "feature_names_in_"):
            print(f"    features: {list(obj.feature_names_in_)}")
        else:
            print(f"    Attrs  : {[a for a in dir(obj) if not a.startswith('_')]}")

print("\n" + "="*55)
print("  Copy the classes_ list above and paste it to Claude")
print("="*55)