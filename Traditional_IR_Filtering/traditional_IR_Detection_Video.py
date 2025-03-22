import cv2
import numpy as np
import multiprocessing as mp
import time
import statistics
import pandas as pd
from skimage.measure import label, regionprops
from skimage.transform import hough_circle, hough_circle_peaks

data_dict = {"Distance": [], "Pixel Counts": [], "Eccentricities": [], "Longest Rows": []
             , "Longest Columns": []}
distance_dict = {}

video_path = "./Videos/video_15_out.avi"
#video_path = "./Videos/video_10metres.mp4"
#video_path = "./Videos/video_20metres_out.avi"

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

def get_pixel_data(image):
    """Gather data about the white pixels from each segmented image."""

    #Counts the number of white pixels
    pixel_count = cv2.countNonZero(image)
    eccentricity = 0.0
    
    #Normalizes the image
    image[image != 0] = 255
    norm_img = image // 255 
    
    #Extracts the eccentricity of each frame
    labeled_image = label(norm_img)
    for region in regionprops(labeled_image):
        eccentricity = region.eccentricity
    
    count = 0
    max_row_count = 0
    max_column_count = 0
    longest_row = []
    longest_column = []
    
    #Extracts the longest uninterrupted horizontal and vertical
    #chain of white pixels from each segmented image frame
    for row in norm_img:
        count = np.count_nonzero(row)
        if count > max_row_count:
            longest_row = row
            max_row_count = count
        for i in range(len(row)):
            if row[i] == 1:
                column = norm_img[:,i]
                count = np.count_nonzero(column)
                if count > max_column_count:
                    longest_column = column
                    max_column_count = count
    
    return (pixel_count, eccentricity, max_row_count, max_column_count)

def get_distance_from_path(video_path):
    """Extracts the distance from the video filename."""

    if "10metres" in video_path:
        return "10m"
    elif "15metres" in video_path or "15_out" in video_path:
        return "15m"
    elif "20metres" in video_path or "20_out" in video_path:
        return "20m"
    return "unknown"  # Default case

    
def store_pixel_data(distance, pix_counts, eccentricities, longest_rows, 
                     longest_columns):
    """Store the data collected from the segmented image frames."""
    #Populates distance_dict
    distance_dict["Distance"] = distance
    distance_dict["Pixel Counts"] = pix_counts
    distance_dict["Eccentricities"] = eccentricities
    distance_dict["Longest Rows"] = longest_rows
    distance_dict["Longest Columns"] = longest_columns

    #Store the data to a dictionary to be written to the csv
    data_dict["Distance"].append(distance)
    data_dict["Pixel Counts"].append(pix_counts)
    data_dict["Eccentricities"].append(eccentricities)
    data_dict["Longest Rows"].append(longest_rows)
    data_dict["Longest Columns"].append(longest_columns)


def write_to_csv(distance, data_dict):
    """Writes the collected data to a csv file."""

    df = pd.DataFrame(data_dict)
    #Creates the csv file containing data for the starting distance of 10m
    if distance == "10m":
        df.to_csv('img_data.csv', index=False)
    #Appends to the csv file instead of overwriting if the distance is not 10m
    else:
        #Check if the row is already in the file
        existing_file = pd.read_csv('img_data.csv')
        existing_data = existing_file.to_dict(orient='records')

        #Writes to the csv if the row does not exist
        write_condition = distance_dict not in existing_data
        print(write_condition)
        if (write_condition):
            df.to_csv('img_data.csv', mode='a', index=False, header=False)
    
    

def main():
    """Reads frames and processes them using multiprocessing with buffering."""
    frame_queue = mp.Queue(maxsize=FRAME_BUFFER_SIZE)
    segmented_queue = mp.Queue(maxsize=FRAME_BUFFER_SIZE)
    processing_times = []
    pool = mp.Pool(processes=MAX_WORKERS)
    
    # Start display process
    display_process = mp.Process(target=display_frames, args=(frame_queue, segmented_queue), daemon=True)
    display_process.start()
    
    pixel_counts = "" #String of white pixel counts from each 
    longest_rows = ""
    longest_columns = ""
    eccentricities = "" #String of eccentricities from each segmented image frame.
    

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
                pixel_counts += str(get_pixel_data(segmented_image)[0]) + ' '
                eccentricities += str(get_pixel_data(segmented_image)[1]) + ' '
                longest_rows += str(get_pixel_data(segmented_image)[2]) + ' '
                longest_columns += str(get_pixel_data(segmented_image)[3]) + ' '
                frame_queue.put(frame_display)
                segmented_queue.put(segmented_image)
                processing_times.append(processing_time)
            results = results[FRAME_BUFFER_SIZE:]
    
    # Process remaining frames
    for res in results:
        frame_display, segmented_image, processing_time = res.get()
        pixel_counts + str(get_pixel_data(segmented_image)[0]) + ' '
        eccentricities + str(get_pixel_data(segmented_image)[1]) + ' '
        longest_rows + str(get_pixel_data(segmented_image)[2]) + ' '
        longest_columns + str(get_pixel_data(segmented_image)[3]) + ' '
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

    distance = get_distance_from_path(video_path)

    #Store the pixel data and write it to a csv file
    store_pixel_data(distance, pixel_counts, eccentricities, longest_rows, longest_columns)
    write_to_csv(distance, data_dict)
    
    
    # Compute and print the average frame processing time
    if processing_times:
        avg_time = statistics.mean(processing_times)
        print(f"Average frame processing time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    main()
