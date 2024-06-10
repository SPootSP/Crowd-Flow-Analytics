from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

# Test Script - Not used much: This script compares the processing time of performing inference on compressed videos, using imgsz and compressing beforehand

# Initialize model
model = YOLO("yolov8x-pose-p6.pt")

# Initialize parameters
count_results = []
processing_times = []


# Count the people in the each frame for non-anonimized data

video = cv2.VideoCapture("./data/video (2160p).mp4")

assert video.isOpened(), "Error reading video file"
w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
print(w,h,fps)

start_time = time.time()
counts = []

while video.isOpened():

    success, frame = video.read()

    if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        
    # Perform inteference on blurred frame
    results = model.track(frame, persist=True,
                            imgsz=int(w/2))
        
    # Extract object ID's and count people in the frame
    track_ids = results[0].boxes.id.int().cpu().tolist()
    count = len(track_ids)
    counts.append(count)
        
        
# Save the results of counting in the video
count_results.append(counts)

# Release and close the video
video.release()

processing_time = time.time() - start_time
processing_times.append(processing_time)

video2 = cv2.VideoCapture("./data/video (2160p)-half.mp4")
w2, h2, fps2 = (int(video2.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
assert video2.isOpened(), "Error reading video file"
while video2.isOpened():

    success, frame = video.read()

    if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        
    # Perform inteference on blurred frame
    results = model.track(frame, persist=True,
                            imgsz=int(w2))
        
    # Extract object ID's and count people in the frame
    track_ids = results[0].boxes.id.int().cpu().tolist()
    count = len(track_ids)
    counts.append(count)
        
        
# Save the results of counting in the video
count_results.append(counts)

# Release and close the video
video2.release()
cv2.destroyAllWindows()
processing_time = time.time() - start_time
processing_times.append(processing_time)


# Define the x range
xrange = range(len(count_results[0]))
plt.plot(xrange, count_results[0])
plt.plot(xrange, count_results[1])

plt.savefig('imgz vs preproc.png')

