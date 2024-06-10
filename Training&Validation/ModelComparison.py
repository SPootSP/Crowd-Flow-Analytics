from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# This script is for comparing the performance of different models
# List your model directories
current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_path = os.path.join(parent_directory,'aies','YoloModels', 'PreTrainedModels')

# Initialize parameters
counts = []
count_results = []
processing_times = []
Models_label = []
model_paths = []

# Retrieve list of model names/paths
for file_name in os.listdir(data_path):
    Models_label.append(file_name)
    model_path =  os.path.join(data_path,file_name)
    model_paths.append(model_path)
model_paths = sorted(model_paths)
Models_label = sorted (Models_label)
print(Models_label)


# Loop over the available models
for i in range(len(Models_label)):
    
    start_time = time.time()

    # Open the video and extract its features
    video = cv2.VideoCapture("./data/video (2160p).mp4")
    assert video.isOpened(), "Error reading video file"
    w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Select the model
    model = YOLO(Models_label[i])

    # Read the video
    while video.isOpened():

        # Read the frame
        success, frame = video.read()

        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        print(Models_label[i])

        # Perform interference 
        results = model.track(frame, persist=True,
                            imgsz=w, classes=0)

        # Extract the tracking ID's and count the objects in the frame 
        while True:
            try:   
                track_ids = results[0].boxes.id.int().cpu().tolist()
                count = len(track_ids)
                counts.append(count)
            except AttributeError:
                count = len(results[0].boxes)

    # Save the results of counting in the video
    count_results.append(counts)

    # Release and close the video
    video.release()
    cv2.destroyAllWindows()

    # Calculate the processing time
    processing_time = time.time() - start_time
    processing_times.append(processing_time)

# Define the x range
xrange = range(len(count_results[0]))

# Setup plot labels
plt.xlabel('frame')
plt.ylabel('amount of people counted')
plt.title('Amount of people in for various models')

# Plot the results
for i in range(len(xrange)):
    plt.plot(xrange, count_results[i], label=Models_label[i])
plt.legend(loc="lower right")

# Save the resuling plot
plt.savefig('model_comp_result.png')
plt.close()

# Plot the processing times and average counts per frame for each model
ave_counts=[]
for list in count_results:
    ave_counts.append(sum(list)/len(list))
plt.xlabel('frame')
plt.ylabel('amount of people counted')
plt.title('Processing time for various models')
plt.bar(Models_label, processing_times)
plt.plot(xrange, ave_counts)
plt.scatter(xrange, ave_counts)
plt.savefig('model_comp_time_vs_counts')