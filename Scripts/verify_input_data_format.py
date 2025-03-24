import numpy as np

DATASET_PATH = "C:/carla_env/fgcu-carla/scripts/preprocessed/combined/standardized_data.npy"

# Load dataset
data = np.load(DATASET_PATH, allow_pickle=True)
print(f"Dataset Shape: {data.shape}")

# Check a sample
sample = data[0]
print(f"Sample 0 shape: {sample.shape if isinstance(sample, np.ndarray) else len(sample)}")
