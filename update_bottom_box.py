import os
import pandas as pd
import numpy as np
import shutil

# Paths
IMG_DIR = "trainingData/hsf_data/hsf_handwritten_boxes"
CSV_PATH = "trainingData/hsf_data/hsf_labels.csv"

splits = {
    "Training": 0.6,
    "Testing": 0.2,
    "Validation": 0.2,
}

out_dirs = {
    "Training": "trainingData/Training/training_words",
    "Testing": "trainingData/Testing/testing_words",
    "Validation": "trainingData/Validation/validation_words",
}

out_csvs = {
    "Training": "trainingData/Training/training_labels.csv",
    "Testing": "trainingData/Testing/testing_labels.csv",
    "Validation": "trainingData/Validation/validation_labels.csv",
}

def main():
    df = pd.read_csv(CSV_PATH)
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    n_train = int(splits["Training"] * n)
    n_test = int(splits["Testing"] * n)
    n_val = n - n_train - n_test

    split_indices = {
        "Training": (0, n_train),
        "Testing": (n_train, n_train + n_test),
        "Validation": (n_train + n_test, n),
    }

    for split, (start, end) in split_indices.items():
        split_df = df.iloc[start:end].copy()
        os.makedirs(out_dirs[split], exist_ok=True)
        for img_name in split_df["IMAGE"]:
            src = os.path.join(IMG_DIR, img_name)
            dst = os.path.join(out_dirs[split], img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        split_df.to_csv(out_csvs[split], index=False)
        print(f"{split}: {len(split_df)} samples, images copied to {out_dirs[split]}, labels to {out_csvs[split]}")

if __name__ == "__main__":
    main()