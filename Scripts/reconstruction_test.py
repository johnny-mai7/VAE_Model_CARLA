import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from train_vae import VAE

# Load dataset
DATASET_PATH = "C:/carla_env/fgcu-carla/scripts/preprocessed/combined/standardized_data.npy"
MODEL_PATH = "C:/carla_env/fgcu-carla/scripts/models/vae_model.pth"

data = np.load(DATASET_PATH, allow_pickle=True)

# Load trained model
input_dim = data.shape[1]
latent_dim = 20
vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
vae.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
vae.eval()

# Select 100 random samples
num_samples = 100
indices = np.random.choice(len(data), num_samples, replace=False)
input_samples = np.array([data[i] for i in indices])

# Convert to tensor
input_tensor = torch.tensor(input_samples, dtype=torch.float32)

# Run through VAE
with torch.no_grad():
    reconstructed_samples, _, _ = vae(input_tensor)

# Compute MSE for each sample
mse_list = [mean_squared_error(input_samples[i], reconstructed_samples[i].detach().numpy()) for i in range(num_samples)]
avg_mse = np.mean(mse_list)

print(f"📉 **Average Reconstruction MSE: {avg_mse:.5f}**")
