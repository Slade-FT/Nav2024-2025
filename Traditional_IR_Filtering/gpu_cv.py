import cv2
import numpy as np

# print(cv2.getBuildInformation())  # Uncomment to check CUDA support

# Check if GPU support is available
if not cv2.cuda.getCudaEnabledDeviceCount():
    print("CUDA-enabled GPU not found. Exiting...")
    exit()

# Load an image using OpenCV
image = cv2.imread("Traditional_IR_Filtering/Images/original.png", cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    print("Error loading image. Exiting...")
    exit()

# Ensure correct dtype (uint8)
image = image.astype("uint8")

# Upload the image to the GPU
gpu_mat = cv2.cuda_GpuMat()
gpu_mat.upload(image)

# Create a Gaussian filter for grayscale images (CV_8UC1)
gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 1.5)

# Apply the filter
gpu_blurred = gaussian_filter.apply(gpu_mat)

# Download result back to CPU
blurred = gpu_blurred.download()

# Save or display
cv2.imshow("Blurred Image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
