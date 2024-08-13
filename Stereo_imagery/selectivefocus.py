import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt



# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image
# ================================================

def getDisparityMap(imL, imR, numDisparities = 64, blockSize = 7, input_type='grayscale'):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    # Preprocess images based on input type
    if input_type == 'edge':
        imL = cv2.Canny(imL, 90, 150)
        imR = cv2.Canny(imR, 90, 150)

    elif input_type == 'gaussian':
        imL = cv2.GaussianBlur(imL, (7, 7), 1)
        imR = cv2.GaussianBlur(imR, (7, 7), 1)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    disparity = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return disparity  # floating point image



import cv2
import numpy as np

def getDisparityMap(imL, imR, numDisparities=64, blockSize=7, input_type='grayscale'):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    # Preprocess images based on input type
    if input_type == 'edge':
        imL = cv2.Canny(imL, 90, 150)
        imR = cv2.Canny(imR, 90, 150)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    disparity = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return disparity  # floating point image

def update_disparity(_):
    global numDisparities, blockSize, k
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize, "gaussian")
    depth = 1 / (disparity + k)
    depth_img = np.interp(depth, (depth.min(), depth.max()), (0, 255)).astype(np.int32)

    # Apply selective focus
    blurred_bg = cv2.GaussianBlur(imgL, (0, 0), sigmaX=5)  # You may adjust sigmaX for the blur effect
    result = np.where(depth_img < 200, imgL, blurred_bg)  # Adjust the threshold as needed
    cv2.imshow('Result', result)

    # Display depth image
    cv2.imshow('Depth', depth_img)

def update_disparity(_):
    global numDisparities, blockSize, k
    disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize, "gaussian")
    depth = 1 / (disparity + k)
    depth_img = np.interp(depth, (depth.min(), depth.max()), (0, 255)).astype(np.float32)
    # depth_img = depth

    # print(disparity[0 ])

    # Apply selective focus
    blurred_bg = cv2.GaussianBlur(imgL, (0, 0), sigmaX=5)  # You may adjust sigmaX for the blur effect
    result = np.where(depth_img < threshold_value, imgL, blurred_bg)  # Adjust the threshold as needed
    cv2.imshow('Result', result)

    # Display depth image
    cv2.imshow('Depth', depth_img)



    

# Load left image


filename_left = 'girlL.png'
imgL = cv2.imread(filename_left, cv2.IMREAD_GRAYSCALE)
if imgL is None:
    print('\nError: failed to open {}.\n'.format(filename_left))
    sys.exit()

# Load right image
filename_right = 'girlR.png'
imgR = cv2.imread(filename_right, cv2.IMREAD_GRAYSCALE)
if imgR is None:
    print('\nError: failed to open {}.\n'.format(filename_right))
    sys.exit()

# Create windows to display the result and depth
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
cv2.moveWindow('Depth', 500,0) 


# Default parameters
numDisparities = 1  # Initial value set to the minimum allowed (4)
blockSize = 7
k = 0
threshold_value = 100

# Create trackbars for parameter adjustment
cv2.createTrackbar('Block Size', 'Result', blockSize, 100, update_disparity)
cv2.createTrackbar('Num Disparities', 'Result', 4, 16, update_disparity)  # Reduce range to [1, 64]
cv2.createTrackbar('k', 'Result', 0, 100, update_disparity)  # Adjust the range as needed
cv2.createTrackbar('Threshold', 'Result', threshold_value, 255, lambda x: None)

while True:
    # Update parameters
    blockSize = cv2.getTrackbarPos('Block Size', 'Result')
    numDisparities = 16 * cv2.getTrackbarPos('Num Disparities', 'Result')  # Scale up by 16
    k = cv2.getTrackbarPos('k', 'Result') / 100.0  # Normalize k between 0 and 1
    threshold_value = cv2.getTrackbarPos('Threshold', 'Result')
    

    # Ensure blockSize is odd
    if blockSize % 2 == 0:
        blockSize += 1

    # Update result and depth
    update_disparity(None)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord(' ') or key == 27:
        break

cv2.destroyAllWindows()

