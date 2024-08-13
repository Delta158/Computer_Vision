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

def getDisparityMap(imL, imR, numDisparities = 64, blockSize = 7, input_type='grayscale', save_path='n'):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    if save_path:
        # Save input images
        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(imL, cmap='gray')
        plt.title('Left Input Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(imR, cmap='gray')
        plt.title('Right Input Image')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('input_images.png')
        plt.close()


    # Preprocess images based on input type
    if input_type == 'edge':
        # imL = cv2.Canny(imL, 25, 45)
        # imR = cv2.Canny(imR, 25, 45)

        imL = cv2.Canny(imL, 90, 150)
        imR = cv2.Canny(imR, 90, 150)
    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1  # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0  # Map is fixed point int with 4 fractional bits

    disparity = cv2.normalize(disparity, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return disparity  # floating point image

# ================================================
#
def plot(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    x = []
    y = []
    z = []
    for r in range(4):
        for c in range(4):
                x += [c]
                y += [r]
                z += [r*c]

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, y, z, 'green')

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()


if __name__ == '__main__':

    imgL = cv2.imread('umbrellaL.png', cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('umbrellaR.png', cv2.IMREAD_GRAYSCALE)

    # Create a window to display the disparity map
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)

    # Default parameters
    numDisparities = 4  # Initial value set to the minimum allowed (4)
    blockSize = 7

    def update_disparity(_):
        global numDisparities, blockSize
        disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize, "edge", "save")
        disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
        cv2.imshow('Disparity', disparityImg)

    # Create trackbars for parameter adjustment
    cv2.createTrackbar('Block Size', 'Disparity', blockSize, 100, update_disparity)
    cv2.createTrackbar('Num Disparities', 'Disparity', numDisparities, 16, update_disparity)

    while True:
        # Update parameters
        blockSize = cv2.getTrackbarPos('Block Size', 'Disparity')
        numDisparities = 16 * cv2.getTrackbarPos('Num Disparities', 'Disparity')  # Scale up by 16

        # Ensure blockSize is odd
        if blockSize % 2 == 0:
            blockSize += 1

        # Update disparity map
        disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize,"edge")
        disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

        # Show result
        cv2.imshow('Disparity', disparityImg)
        
        # Check for key press
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
