import cv2
import numpy as np

# Given parameters
sensor_width_mm = 22.2  # Sensor width in millimeters
sensor_height_orig_mm = 14.8  # Original sensor height in millimeters
sensor_height_resized_mm = (sensor_height_orig_mm * 505) / 740  # Resized sensor height in millimeters
image_width = 740  # Resized image width in pixels
image_height = 505  # Resized image height in pixels
focal_length_cam0_px = 5806.559  # Focal length for cam0 in pixels
focal_length_cam1_px = 5806.559  # Focal length for cam1 in pixels

# Calculate focal length in millimeters
focal_length_cam0_mm = (focal_length_cam0_px * sensor_width_mm) / image_width
focal_length_cam1_mm = (focal_length_cam1_px * sensor_width_mm) / image_width

# Print formatted focal lengths
print("Focal length for cam0: {:.4f} mm".format(focal_length_cam0_mm))
print("Focal length for cam1: {:.4f} mm".format(focal_length_cam1_mm))

# [5806.559 0 1429.219; 0 5806.559 993.403; 0 0 1]

# Camera matrices
cam0 = np.array([[5806.559, 0, 1429.219],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])

cam1 = np.array([[5806.559, 0, 1543.51],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])

# Calculate calibration parameters
c0_small = cv2.calibrationMatrixValues(cam0, (2960, 2016), 22.2, 14.8)
c1_small = cv2.calibrationMatrixValues(cam1, (2960, 2016), 22.2, 14.8)
c0_big = cv2.calibrationMatrixValues(cam0, (3088 , 2056), 22.2, 14.8)
c1_big = cv2.calibrationMatrixValues(cam1, (3088 , 2056), 22.2, 14.8)

print(c0_small)
print(c1_small)
print(c0_big)
print(c1_big)

print(f"cam0 2960 x 2016 focal length: {c0_small[2]:.3f}")
print(f"cam1 2960 x 2016 focal length: {c1_small[2]:.3f}")
print(f"cam0 3088 x 2056 focal length: {c0_big[2]:.3f}")
print(f"cam1 3088 x 2056 focal length: {c1_big[2]:.3f}")

