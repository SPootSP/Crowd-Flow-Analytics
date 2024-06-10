from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os

# This script is for comparing the performance of various models on different levels of blurring. 
# List your model directories and the blurring you want to evaluate with.

current_directory = os.getcwd()
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
data_path = os.path.join(parent_directory,'aies','YoloModels', 'COCOModels')

# Initialize parameters
count_results = []
processing_times = []
blurs = [1,3,5,7,9]
Models_label = []
model_paths = []
video_path = './data/video (2160p).mp4'

# Retrieve list of model names/paths
for file_name in os.listdir(data_path):
    Models_label.append(file_name)
    model_path =  os.path.join(data_path,file_name)
    model_paths.append(model_path)
model_paths = sorted(model_paths)
Models_label = sorted (Models_label)
print(Models_label)

# Go over every model on non-blurred data
for i in range(len(model_paths)):

    # Choose the model
    model = YOLO(model_paths[i])

    # Initialize/reset counts of a frame
    counts = []

    # Open the video
    video = cv2.VideoCapture(video_path)
    assert video.isOpened(), "Error reading video file"

    # Retrieve video parameters
    w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


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
                                     imgsz=w, device="cuda:0", classes=0)
        print(model_paths[i])
        # Extract the tracking ID's and count the objects in the frame and store the results  
        while True:
            try:  
                track_ids = results[0].boxes.id.int().cpu().tolist()
                count = len(track_ids)
                counts.append(count)
                break
            except AttributeError:
                count = len(results[0].boxes)
                print(count)
                counts.append(count)
                break

                

        # Save the results of counting in the video
        count_results.append(counts)

        # Release and close the video
        video.release()
        cv2.destroyAllWindows()

        # Calculate and store the time the process took
        processing_time = time.time() - start_time
        processing_times.append(processing_time)


# Analyze the video for all chosen blur levels
for j in range(len(blurs)):


    # Perform inference over the whole video every model
    for i in range(len(model_paths)):

        # Choose the model
        model = YOLO(model_paths[i])

        # Initialize/reset counts of a frame
        counts = []

        # Open the video
        video = cv2.VideoCapture(video_path)
        assert video.isOpened(), "Error reading video file"

        # Retrieve video parameters
        w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        # Start the time
        start_time = time.time()

        while video.isOpened():

            # Read the frame
            success, frame = video.read()
            
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            
            # Blur the frame
            frame = cv2.GaussianBlur(frame, (0,0), blurs[j])
            
            # Perform interference 
            results = model.track(frame, persist=True,
                                        imgsz=w, device="cuda:0", classes=0)
            print(model_paths[i])
            # Extract the tracking ID's and count the objects in the frame and store the results  
            while True:
                try:  
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    count = len(track_ids)
                    counts.append(count)
                    break
                except AttributeError:
                    count = len(results[0].boxes)
                    print(count)
                    counts.append(count)
                    break

                

        # Save the results of counting in the video
        count_results.append(counts)

        # Release and close the video
        video.release()
        cv2.destroyAllWindows()

        # Calculate and store the time the process took
        processing_time = time.time() - start_time
        processing_times.append(processing_time)

#setup the plot

# Calculate the average amount of people in a frame
ave_counts=[]
for list in count_results:
    ave_counts.append(sum(list)/len(list))
blurs_label = [0,1,3,5,7,9]
ave_array = np.array(ave_counts).reshape((len(blurs_label),len(model_paths)))
ave_array = np.transpose(ave_array)

# Set up the plot
plt.title('Models used on different blurs')

plt.matshow(ave_array)
plt.colorbar()
plt.xticks(range(len(blurs_label)), blurs_label)
plt.yticks(range(len(model_paths)), Models_label)
for (x,y), value in np.ndenumerate(np.transpose(ave_array)):
    plt.text(x, y, f'{value:.2f}', va='center', ha='center', fontsize = 7, bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

plt.savefig('matrix plot models vs blur-coco',bbox_inches='tight')
plt.close()

print(processing_times,count_results,ave_counts,ave_array)




