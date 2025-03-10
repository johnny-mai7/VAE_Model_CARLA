# 🚗 VAE-Based OOD Detection for Autonomous Driving

https://github.com/carla-simulator/carla
---
*A deep learning approach to Out-of-Distribution (OOD) detection in self-driving systems using multi-modal sensor data.*

---

## 📌 Overview
This project leverages a **Variational Autoencoder (VAE)** to detect **Out-of-Distribution (OOD) data** in an autonomous driving scenario simulated in **CARLA**. The system integrates **camera, LiDAR, radar, and vehicle state data**, preprocesses them, and trains a VAE model to distinguish between normal and anomalous driving conditions. If OOD data is detected, the **vehicle switches to manual driving mode** to prevent failures.

---

## 📂 Project Structure

---

📁 Scripts 
---
  collect_data.py # Collects multi-modal sensor data from CARLA 

  preprocess_images.py # Converts and normalizes camera images into .npy format 

  ensure_sequential_filenames.py # Ensures all files are numbered correctly 

  combine_multimodal_data.py # Merges all sensor modalities into a single dataset 

  train_vae.py # Trains the Variational Autoencoder 

  reconstruction_test.py # Evaluates model reconstruction performance 

  debug_file_check.py # Checks dataset integrity across all levels 

  missing_data.py # Identifies missing samples in each level 

📁 Preprocessed 
---

 📁 camera_npy/ # Preprocessed camera images in .npy format 

 📁 lidar/ # Processed LiDAR data

 📁 radar/ # Processed radar data 

 📁 vehicle_state/ # Processed vehicle state logs

 combined_data.npy # Final multi-modal dataset for training 

📁 Models 
---
vae_model.pth # Trained VAE model 

  README.md
