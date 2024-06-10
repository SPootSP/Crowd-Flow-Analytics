from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
from datetime import datetime
import time

# The script is for analyzing the effect of skipping frames on the total processing time and accuracy of the inference.

# Initialize parameters
counts = []
count_results = []
frame_skips = [2,3,5,10]
processing_times = []
time_stamps = []
total_frames = []
frame_count = 1
data_path = "./data/madras-poc.mp4"
model_path = './YoloModels/PreTrainedModels/yolov8x.pt'

# Initialize model
model = YOLO(model_path)

# Count the people in the each frame without skipping frames

# Open the video
video = cv2.VideoCapture(data_path)
assert video.isOpened(), "Error reading video file"

# Retrieve video parameters
w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(fps)

# Start the time
start_time = time.time()

# Read the video
while video.isOpened():

    # Read the frame
    success, frame = video.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    
    
    # Perform interference 
    results = model.track(frame, persist=True,
                                imgsz=w, classes = 0)

    # Extract the tracking ID's and count the objects in the frame  
    while True:
        try:  
            track_ids = results[0].boxes.id.int().cpu().tolist()
            count = len(track_ids)
            counts.append(count)
            break
        except AttributeError:
            count = len(results[0].boxes)
            counts.append(count)
            break

        

# Save the results of counting in the video
count_results.append(counts)
total_frames.append(len(count_results[0]))

# Release and close the video
video.release()
cv2.destroyAllWindows()

# Stop the time and calculate the elapsed time.
processing_time = time.time() - start_time
processing_times.append(processing_time)


# Count the people in the each frame for different amounts of frames skipped

for i in range(len(frame_skips)):

    # Start the time
    start_time = time.time()

    frame_count = 1

    # Open the video and retrieve parameters
    video = cv2.VideoCapture(data_path)
    assert video.isOpened(), "Error reading video file"
    w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    counts = []

    # Open the video
    while video.isOpened():

        # Read the frame
        success, frame = video.read()

        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        # Perform inference if the frame is frames are skipped (check if the remainder of division is zero)
        if frame_count % frame_skips[i] == 0:
            
            
            # Perform inteference on blurred frame
            results = model.track(frame, persist=True,
                                imgsz=w, classes = 0)
            
            # Extract object ID's and count people in the frame
            while True:
                try:  
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    count = len(track_ids)
                    counts.append(count)
                    break
                except AttributeError:
                    count = len(results[0].boxes)
                    counts.append(count)
                    break
            
        # Count the frame
        frame_count += 1

        
    # Save the results of counting in the video
    count_results.append(counts)

    # Release and close the video
    video.release()
    cv2.destroyAllWindows()

    # Calculate the processing time
    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    total_frames.append(len(count_results[i+1]))

# Release and close the video
cv2.destroyAllWindows()

#setup the plot
# Define the x range
xrange = []
xrange.append(fps)
for x in frame_skips:
    xrange.append(fps/x)
print(xrange)

# Setup plot labels
plt.xlabel('fps log(1/s)')
plt.ylabel('processing time log(s)')
plt.title('processing time depending on the fps')

# Plot the results
plt.plot(xrange,processing_times)
plt.scatter(xrange,processing_times)
plt.xscale('log')
plt.yscale('log')
#plt.legend(loc='upper left')

# Save the resuling plot
plt.savefig('frame rate results.png')
plt.close()

# Determine the average amount of people per frame in the video.
ave_counts=[]
for list in count_results:
    ave_counts.append(sum(list)/len(list))

# Setup plot labels
plt.ylabel('average amount of people counted')
plt.xlabel('frames skipped')
plt.title('The average amount of people counted depending on the fps')

# Plot the figure
plt.plot(xrange, ave_counts)
plt.scatter(xrange, ave_counts)
plt.savefig('frame rate vs ave counts.png')
plt.close()


# Compare with build in frame skipping tool from ultralytics

# Initialize parameters
count_results2 = []
processing_times2 = []
frame_skips2 = [1,2,3,5,10]

# Use the build in frame skipping tool for comparison
for i in range(len(frame_skips2)):

    # Start the time
    start_time = time.time()

    counts = []     
        
    # Perform inteference
    results = model.track(data_path, persist=True,
                                imgsz=w, vid_stride=frame_skips2[i], classes = 0)

    # Calculate the processing time
    processing_time = time.time() - start_time
    processing_times2.append(processing_time)


# Setup the plot
# Define the x range
xrange = []
xrange.append(fps)
for x in frame_skips:
    xrange.append(fps/x)
print(xrange, len(xrange), len(processing_times2))

# Setup plot labels
plt.xlabel('fps Log(1/s)')
plt.ylabel('processing time log (s)')
plt.title('processing time depending on the fps')

# Plot the results
plt.plot(xrange,processing_times, label='OpenCv')
plt.plot(xrange,processing_times2, label='Ultralytics-build in')
plt.scatter(xrange,processing_times)
plt.scatter(xrange,processing_times2)
plt.legend(loc='upper right')
plt.xscale('log')
plt.yscale('log')

# Save the resuling plot
plt.savefig('frame rate results-compare.png')
plt.close()