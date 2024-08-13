import cv2
import numpy as np

def sobel_operator(image):
    # Sobel operator for x and y derivatives
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return sobel_x, sobel_y

def gaussian_kernel(size, sigma):
    kernel_1d = cv2.getGaussianKernel(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d.T)
    kernel_2d /= kernel_2d.sum()
    return kernel_2d

def compute_harris_matrix(image, k_size=5, sigma=0.5):
    sobel_x, sobel_y = sobel_operator(image)
    height, width = image.shape

    # Padding the image
    padded_image = cv2.copyMakeBorder(image, k_size//2, k_size//2, k_size//2, k_size//2, cv2.BORDER_REFLECT)

    # Compute Harris matrix M
    M_list = []
    for y in range(height):
        for x in range(width):
            window_x = sobel_x[y:y+k_size, x:x+k_size]
            window_y = sobel_y[y:y+k_size, x:x+k_size]

            # Compute weighted sums
            weight = gaussian_kernel(k_size, sigma)
            Ix2 = np.sum(window_x * window_x * weight)
            Ixy = np.sum(window_x * window_y * weight)
            Iy2 = np.sum(window_y * window_y * weight)

            # Construct Harris matrix M
            M = np.array([[Ix2, Ixy],
                          [Ixy, Iy2]])
            M_list.append((x, y, M))

    return M_list

def compute_corner_strength(M_list):
    R_list = []
    for x, y, M in M_list:
        # Compute corner strength function R
        det_M = np.linalg.det(M)
        trace_M = np.trace(M)
        R = det_M - 0.05 * (trace_M ** 2)
        R_list.append((x, y, R))
    return R_list

def find_local_maxima(R_list, neighborhood_size=7):
    local_maxima = []
    height, width = image.shape

    # Look for local maxima in the R list
    for x, y, R in R_list:
        # Check if (x, y) is the maximum in its neighborhood
        neighborhood = R_list[max(0, x - neighborhood_size//2):min(width, x + neighborhood_size//2 + 1),
                              max(0, y - neighborhood_size//2):min(height, y + neighborhood_size//2 + 1), 2]
        if np.all(R >= neighborhood):
            local_maxima.append((x, y, R))

    return local_maxima

# Load image
image = cv2.imread('bernieSanders.jpg', cv2.IMREAD_GRAYSCALE)

# Compute Harris matrix
M_list = compute_harris_matrix(image)

# Compute corner strength
R_list = compute_corner_strength(M_list)

# Find local maxima
local_maxima = find_local_maxima(R_list)

# Print local maxima
print(local_maxima)
