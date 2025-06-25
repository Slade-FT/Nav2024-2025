import cv2
import numpy as np
import multiprocessing as mp
import time
import statistics
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

video_path = "./Traditional_IR_Filtering/Videos/video_15_out.avi"
#video_path = "./Traditional_IR_Filtering/Videos/video_10metres.mp4"
#video_path = "./Traditional_IR_Filtering/Videos/video_20metres_out.avi"

cap = cv2.VideoCapture(video_path)

# Multiprocessing Parameters
MAX_WORKERS = mp.cpu_count()
FRAME_BUFFER_SIZE = 30  # Buffer size for processed frames


def process_frame(frame):
    """Processes a single frame, detects circles, segments the most circular object,
    and extracts pixel values from it."""
    
    start_time = time.time()
    
    # Resize frame to speed up processing
    frame = cv2.resize(frame, (640, 480))
    
    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame_gray_norm = frame_gray / 255.0  # Normalize
    
    # thresholding
    binary_image = frame_gray_norm > 0.95
    
    # Remove large non-circular areas
    labeled_img = label(binary_image)
    mask = np.zeros_like(binary_image, dtype=bool)
    
    for region in regionprops(labeled_img):
        if 10 <= region.area <= 500 and region.eccentricity < 0.68:
            mask[labeled_img == region.label] = True
    
    # Apply Canny Edge Detection
    edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
    
    # Detect circles using Hough Transform
    hough_radii = np.arange(15, 100, 5)  # Search for circles in this range
    hough_res = hough_circle(edges, hough_radii)
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)
    
    # Find the most circular object
    most_circular_idx = None
    lowest_eccentricity = float("inf")
    
    for i, (x, y, r) in enumerate(zip(cx, cy, radii)):
        region_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(region_mask, (x, y), r, 1, -1)
        labeled_mask = label(region_mask)
        regions = regionprops(labeled_mask)
        
        if regions:
            eccentricity = regions[0].eccentricity
            if eccentricity < lowest_eccentricity:
                lowest_eccentricity = eccentricity
                most_circular_idx = i
    
    # Prepare output images
    frame_display = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # Convert grayscale to color for visualization
    segmented_image = np.zeros_like(frame_gray)  # Black background for segmented region
    
    if most_circular_idx is not None:
        x, y, r = cx[most_circular_idx], cy[most_circular_idx], radii[most_circular_idx]
        
        # Create a circular mask for the segmented region
        circle_mask = np.zeros_like(frame_gray, dtype=np.uint8)
        cv2.circle(circle_mask, (x, y), r, 255, -1)

        # Create a blank grayscale image for visualization
        segmented_image = np.zeros_like(frame_gray)

        # Apply threshold (keep only values > 230)
        threshold_mask = (frame_gray > 230) & (circle_mask == 255)

        # Apply the threshold to the segmented image
        segmented_image[threshold_mask] = frame_gray[threshold_mask]
        
        # Draw the detected circle on the frame display
        cv2.circle(frame_display, (x, y), r, (0, 255, 0), 2)  # Green circle
    
    end_time = time.time()
    return (frame_display, segmented_image, end_time - start_time)


def display_frames(frame_queue, segmented_queue):
    """Continuously displays processed and segmented frames."""
    while True:
        frame_display = frame_queue.get()
        segmented_image = segmented_queue.get()
        
        if frame_display is None or segmented_image is None:
            break  # Stop if sentinel value received
        
        cv2.imshow("Detected Target", frame_display)
        cv2.imshow("Segmented Image", segmented_image)
        
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cv2.destroyAllWindows()


def main():
    """Reads frames and processes them using multiprocessing with buffering."""
    
    frame_queue = mp.Queue(maxsize=FRAME_BUFFER_SIZE)
    segmented_queue = mp.Queue(maxsize=FRAME_BUFFER_SIZE)
    processing_times = []
    pool = mp.Pool(processes=MAX_WORKERS)
    
    # Start display process
    display_process = mp.Process(target=display_frames, args=(frame_queue, segmented_queue), daemon=True)
    display_process.start()
    
    results = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 2nd frame to improve performance
        if frame_count % 2 == 0:
            result = pool.apply_async(process_frame, (frame,))
            results.append(result)
        frame_count += 1
        
        if len(results) >= FRAME_BUFFER_SIZE:
            for res in results[:FRAME_BUFFER_SIZE]:
                frame_display, segmented_image, processing_time = res.get()
                frame_queue.put(frame_display)
                segmented_queue.put(segmented_image)
                processing_times.append(processing_time)
            results = results[FRAME_BUFFER_SIZE:]
    
    # Process remaining frames
    for res in results:
        frame_display, segmented_image, processing_time = res.get()
        frame_queue.put(frame_display)
        segmented_queue.put(segmented_image)
        processing_times.append(processing_time)
    
    # Stop the display process
    frame_queue.put(None)
    segmented_queue.put(None)
    display_process.join()
    
    cap.release()
    pool.close()
    pool.join()
    
    # Compute and print the average frame processing time
    if processing_times:
        avg_time = statistics.mean(processing_times)
        print(f"Average frame processing time: {avg_time:.4f} seconds")


if __name__ == "__main__":
    main()
