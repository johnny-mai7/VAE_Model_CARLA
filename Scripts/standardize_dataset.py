import numpy as np

# Define dataset path
DATASET_PATH = "C:/carla_env/fgcu-carla/scripts/preprocessed/combined/combined_data.npy"
OUTPUT_PATH = "C:/carla_env/fgcu-carla/scripts/preprocessed/combined/standardized_data.npy"

# Load dataset
data = np.load(DATASET_PATH, allow_pickle=True)

# Determine max sample length
max_length = max(len(sample) for sample in data)
print(f"Max sample length: {max_length}")

# Function to pad or truncate samples
def standardize_sample(sample, length):
    if len(sample) < length:
        # Pad with zeros
        return np.pad(sample, (0, length - len(sample)), mode='constant')
    else:
        # Truncate excess
        return sample[:length]

# Standardize all samples
standardized_data = np.array([standardize_sample(sample, max_length) for sample in data])

# Save standardized dataset
np.save(OUTPUT_PATH, standardized_data)
print(f"✅ Standardized dataset saved: {OUTPUT_PATH}")
