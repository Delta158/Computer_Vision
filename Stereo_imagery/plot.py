import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


def plot(disparity, baseline, focal_length, doffs):
    # Initialize arrays to store world coordinates
    x = []
    y = []
    z = []

    # Loop through disparity map
    for r in range(disparity.shape[0]):
        for c in range(disparity.shape[1]):
            # Calculate depth (Z) using the formula
            Z = (baseline * focal_length) / (disparity[r, c] + doffs)

            # Calculate world coordinates (X, Y, Z) using similar triangles and pixel coordinates
            X = c * Z / focal_length
            Y = r * Z / focal_length
            x.append(X)
            y.append(Y)
            z.append(Z)

    # Convert lists to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Plot only non maximum depth points
    max_depth = np.max(z)
    mask = z < max_depth
    x = x[mask]
    y = y[mask]
    z = z[mask]


    # Plot 3D reconstruction from the disparity map
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, z, y, c='g', s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scene')
    ax.view_init(elev=15, azim=240)
    plt.tight_layout()
    plt.savefig("3d.png")
    plt.show()

    # Plot 2D view from above (X, Z)
    plt.figure(figsize=(10, 10))
    plt.scatter(x, z, c='r', s=10)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('2D View from Above (XZ)')
    plt.ylim([z.min(), z.max()])
    plt.tight_layout()
    plt.savefig("top.png")
    plt.show()


    # Plot 2D view from side (Y, Z)
    plt.figure(figsize=(10, 10))
    plt.gca().invert_yaxis()
    plt.scatter(z, y, c='b', s=10)
    plt.xlabel('Z')
    plt.ylabel('Y')
    plt.title('2D View from Side (YZ)')
    plt.xlim([z.min(), z.max()])
    plt.tight_layout()
    plt.savefig("side.png")
    plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt
from disparity import getDisparityMap

# Load left and right images
filename_left = 'umbrellaL.png'
filename_right = 'umbrellaR.png'

# filename_left = 'girlL.png'
# filename_right = 'girlR.png'


imgL = cv2.imread(filename_left, cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(filename_right, cv2.IMREAD_GRAYSCALE)

# Parameters (adjust as needed)
baseline = 174.019  # mm
focal_length = 5806.559  # pixels
doffs = 114.291  # pixels

# focal_length = 1536.318 # mm
# doffs = 30.239 # mm

numDisparities = 64
blockSize = 7

# Compute disparity map
disparity = getDisparityMap(imgL, imgR, numDisparities, blockSize, "edge")

# Plot the scene
plot(disparity, baseline, focal_length, doffs)


