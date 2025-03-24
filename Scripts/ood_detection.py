import carla
import torch
import numpy as np
import pygame
import cv2
import time
from collections import deque
from train_vae import VAE

# 🚗 Initialize PyGame for Full-Screen First-Person View
pygame.init()
INFO = pygame.display.Info()
WIDTH, HEIGHT = INFO.current_w, INFO.current_h  # Fullscreen
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("CARLA First-Person View")

# ⏱️ Timer for FPS control
clock = pygame.time.Clock()

# 🌍 Load Town03 Map
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
client.load_world("Town03")
world = client.get_world()

# 🌧️ Define Weather Levels
WEATHER_PRESETS = [
    carla.WeatherParameters.ClearNoon,
    carla.WeatherParameters.MidRainSunset,
    carla.WeatherParameters.HardRainSunset,
    carla.WeatherParameters.HardRainNoon,
    carla.WeatherParameters.MidRainyNight,
    carla.WeatherParameters.HardRainNight
]
weather_level = 0
manual_mode = False
permanent_manual_mode = False  # 🚨 Ensures manual mode stays if MSE remains high

# WASD Controls for Manual Mode
throttle, brake, steer = 0.0, 0.0, 0.0  

# 🚗 **Spawn Vehicle First**
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)  # ✅ Auto-driving is ON by default

# ✅ **Set Weather AFTER Vehicle is Spawned**
def set_weather(level):
    global world, manual_mode, permanent_manual_mode

    world.set_weather(WEATHER_PRESETS[level])
    print(f"🌦️ Weather changed to Level {level}")

    # 🚨 If level is 3 or above, immediately switch to manual mode
    if level >= 3:
        print("🚨 Severe weather detected! Forcing manual mode!")
        permanent_manual_mode = True
        manual_mode = True
        vehicle.set_autopilot(False)

    # ✅ If user switches back to levels 0, 1, or 2, re-enable auto mode
    elif level <= 2:
        print("✅ Safe weather detected! Re-enabling auto-driving!")
        permanent_manual_mode = False  
        manual_mode = False  
        vehicle.set_autopilot(True)

set_weather(weather_level)  # ✅ Now it's safe to call

# Load Trained VAE Model
MODEL_PATH = "C:/carla_env/fgcu-carla/scripts/models/vae_model.pth"
BASE_MSE_THRESHOLD = 20.0  
INPUT_DIM = 80619
LATENT_DIM = 20

vae = VAE(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
vae.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
vae.eval()

# Attach Camera Sensor
DISPLAY_RES = (1920, 1080)  
VAE_INPUT_SIZE = (128, 128)  

camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", str(DISPLAY_RES[0]))  
camera_bp.set_attribute("image_size_y", str(DISPLAY_RES[1]))  
camera_bp.set_attribute("fov", "110")
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Attach LiDAR Sensor
lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
lidar_bp.set_attribute("range", "50")
lidar_bp.set_attribute("channels", "32")
lidar_bp.set_attribute("points_per_second", "100000")
lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# Attach Radar Sensor
radar_bp = blueprint_library.find("sensor.other.radar")
radar_bp.set_attribute("horizontal_fov", "30")
radar_bp.set_attribute("vertical_fov", "10")
radar_bp.set_attribute("range", "20")
radar_transform = carla.Transform(carla.Location(x=1.5, z=1.0))
radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

latest_display_image, latest_vae_image, latest_lidar, latest_radar = None, None, None, None  

MSE_HISTORY = deque(maxlen=10)  
HIGH_MSE_COUNT = 0  
LOW_MSE_COUNT = 0   

def camera_callback(image):
    global latest_display_image, latest_vae_image

    img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
    img_array = img_array.reshape((image.height, image.width, 4))[:, :, :3]

    latest_display_image = img_array
    latest_vae_image = cv2.resize(img_array, (128, 128)).astype(np.float32) / 255.0

def lidar_callback(lidar_data):
    global latest_lidar
    latest_lidar = lidar_data

def radar_callback(radar_data):
    global latest_radar
    latest_radar = radar_data

camera.listen(camera_callback)
lidar.listen(lidar_callback)
radar.listen(radar_callback)

def preprocess_lidar(lidar_data):
    points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    return np.pad(points, ((0, max(0, 1000 - points.shape[0])), (0, 0)), mode="constant").flatten()

def preprocess_radar(radar_data):
    points = np.zeros((100, 4), dtype=np.float32)
    if radar_data:
        actual_points = np.array([(d.depth, d.velocity, d.azimuth, d.altitude) for d in radar_data], dtype=np.float32)
        points[:actual_points.shape[0]] = actual_points[:100]
    return points.flatten()

def handle_manual_controls():
    global throttle, brake, steer

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w]:  
        throttle = min(throttle + 0.05, 0.5)  # 🚀 Set speed limit
    else:
        throttle = max(throttle - 0.05, 0.0)

    if keys[pygame.K_s]:  
        brake = min(brake + 0.1, 1.0)
    else:
        brake = max(brake - 0.05, 0.0)

    if keys[pygame.K_d]:  
        steer = max(steer - 0.05, -0.5)  
    elif keys[pygame.K_a]:  
        steer = min(steer + 0.05, 0.5)  
    else:
        steer *= 0.9  

    control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steer)
    vehicle.apply_control(control)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if pygame.K_1 <= event.key <= pygame.K_6:
                weather_level = event.key - pygame.K_1
                set_weather(weather_level)

    if latest_display_image is not None:
        surface = pygame.surfarray.make_surface(np.rot90(latest_display_image))
        screen.blit(pygame.transform.smoothscale(surface, (WIDTH, HEIGHT)), (0, 0))
        pygame.display.flip()

    if manual_mode:
        handle_manual_controls()

    clock.tick(30)

pygame.quit()
camera.stop()
lidar.stop()
radar.stop()
vehicle.destroy()
