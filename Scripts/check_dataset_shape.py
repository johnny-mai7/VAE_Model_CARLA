import numpy as np

DATASET_PATH = "C:/carla_env/fgcu-carla/scripts/preprocessed/combined/standardized_data.npy"

# Load dataset
data = np.load(DATASET_PATH, allow_pickle=True)
print(f"Dataset Shape: {data.shape}")

# Inspect first sample
sample_0 = data[0]
print(f"Sample 0 shape: {sample_0.shape if isinstance(sample_0, np.ndarray) else len(sample_0)}")
