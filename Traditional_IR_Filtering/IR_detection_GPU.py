import cv2
import numpy as np
import time
from skimage.measure import label, regionprops

# Check if GPU support is available
if not cv2.cuda.getCudaEnabledDeviceCount():
    print("CUDA-enabled GPU not found. Exiting...")
    exit()

video_path = "./Traditional_IR_Filtering/Videos/video_15_out.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Create CUDA Stream for faster execution
stream = cv2.cuda.Stream()
gpu_mat = cv2.cuda_GpuMat()

def process_frame(frame):
    """Processes a single frame, detects the most circular object."""
    start_time = time.time()

    # Upload frame to GPU asynchronously
    gpu_mat.upload(frame, stream)

    # Resize frame (CUDA)
    gpu_resized = cv2.cuda.resize(gpu_mat, (640, 480), stream=stream)

    # Convert to grayscale (CUDA)
    gpu_gray = cv2.cuda.cvtColor(gpu_resized, cv2.COLOR_BGR2GRAY, stream=stream)

    # Ensure the image is in 8-bit format before thresholding
    gpu_gray.convertTo(cv2.CV_8UC1, stream=stream)

    # Apply thresholding (CUDA)
    _, gpu_binary = cv2.cuda.threshold(gpu_gray, 235, 255, cv2.THRESH_BINARY, stream=stream)

    # Wait for GPU operations to complete
    stream.waitForCompletion()

    # Download the binary image to CPU
    binary_image = gpu_binary.download()

    # Label connected components
    labeled_img = label(binary_image)

    # Find the most circular object (lowest eccentricity)
    best_region = None
    min_eccentricity = float("inf")

    for region in regionprops(labeled_img):
        if 10 <= region.area <= 500 and region.eccentricity < 0.7:  # Ignore very small regions (noise)
            if region.eccentricity < min_eccentricity:
                min_eccentricity = region.eccentricity
                best_region = region

    # Create a mask of the most circular object
    mask = np.zeros_like(binary_image, dtype=np.uint8)
    if best_region:
        mask[labeled_img == best_region.label] = 255

    end_time = time.time()
    return mask, end_time - start_time

# Read and process frames
times_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, processing_time = process_frame(frame)
    times_data.append(processing_time)

    # Display results
    cv2.imshow("Processed Frame", processed_frame)
    cv2.imshow("Original Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Average Frame Processed in {sum(times_data)/len(times_data):.4f} seconds")
cap.release()
cv2.destroyAllWindows()
