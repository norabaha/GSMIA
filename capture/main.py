from picamera2 import Picamera2
import numpy as np
import time
import os
import json
import sys

nr_images = 1              # Number of images to capture
delay_s = 3                 # Delay in seconds between image captures
name="multi_"
image_path = "capture/lightbox" # Path to folder to store images
log_file = "capture/lightbox/log.json"

os.makedirs(image_path, exist_ok=True)

try:
    picam2 = Picamera2()
    if not picam2.global_camera_info():
        print("No cameras detected. Please check the connection and try again.")
        sys.exit(1)
except Exception as e:
    print(f"Error initializing Picamera2: {e}")
    sys.exit(1)

config = picam2.create_still_configuration(raw={
    "size": (2304, 1296),
    "format": "SRGGB10"})
picam2.configure(config)
picam2.start()
time.sleep(delay_s)

# Get current camera controls and configuration
controls = picam2.camera_controls
config = picam2.camera_configuration()['raw']

log_data = {
    "controls": controls,
    "configuration": config
}

# Make sure log directory exists
os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

# Write to log file as JSON
with open(log_file, "w") as f:
    json.dump(log_data, f, indent=4)

print(f"Camera controls and configuration logged to {log_file}")

for i in range(nr_images):
    print(f"Capture image: {i}")
    raw = picam2.capture_array("raw")
    raw16 = raw.view(np.uint16)  # the real 10-bit data

    try:
        filename = os.path.join(image_path, f"img_{name}_{i}.npy")
        np.save(filename, raw16)

        metadata = picam2.capture_metadata()  # dictionary
        log_filename = os.path.join(image_path, f"log_{name}_{i}.json")

        with open(log_filename, "w") as f:
            json.dump(metadata, f, indent=4)

    except Exception as e:
        print(f"Error saving image {i}: {e}")

    time.sleep(delay_s)

picam2.stop()
