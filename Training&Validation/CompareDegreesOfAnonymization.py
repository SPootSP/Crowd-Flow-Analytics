from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import matplotlib.pyplot as plt
import numpy as np

# This script is for testing a model on data with various degrees of anonymization (through blurring).

# Set the relevant pahts
data_path = "./data/madras-poc.mp4"
model_path = './YoloModels/PreTrainedModels/yolov8x-pose-p6.pt'

# Initialize model
model = YOLO(model_path)

# Initialize parameters
counts = []
count_results = []

# Choose the degrees of blurring
sigma = [2,4,6,8]

# Count the people in the each frame for non-anonimized data
# Open the video and retrieve video parameters
video = cv2.VideoCapture(data_path)
assert video.isOpened(), "Error reading video file"
w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


# While the vidoe is open
while video.isOpened():

    # Read the frame
    success, frame = video.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform interference 
    results = model.track(frame, persist=True,
                            imgsz=w, classes = 0)

    # Extract the tracking ID's and count the objects in the frame.
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

# Release and close the video
video.release()
cv2.destroyAllWindows()


# Count the people in the each frame for different stages of anonimized data

for i in range(len(sigma)):

    # Open the video and retrieve parameters
    video = cv2.VideoCapture(data_path)
    assert video.isOpened(), "Error reading video file"
    w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    counts = []

    # Read the vidoe
    while video.isOpened():

        # Read the frame
        success, frame = video.read()

        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        # Blur the frame
        frame = cv2.GaussianBlur(frame, (0,0),sigma[i])
        
        
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
        
    # Save the results of counting in the video
    count_results.append(counts)

    # Release and close the video
    video.release()
    cv2.destroyAllWindows()

# Create an image comparing the blur levels
# Open the video and retrieve parameters
video = cv2.VideoCapture("./data/video (2160p).mp4")
assert video.isOpened(), "Error reading video file"
w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Select a frame
video.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame_anom = video.read()

# The first image is not blurred
img_anom = frame_anom

for i in range(len(sigma)):

    # Select a frame
    video.set(cv2.CAP_PROP_POS_FRAMES, 100)
    success, frame_anom = video.read()

    # Blur the image
    frame_anom = cv2.GaussianBlur(frame_anom, (0,0),sigma[i])

    # Store in a single image
    img_anom = np.concatenate((img_anom,frame_anom), axis=0)

# Save the image
cv2.imwrite('Visual Comparison of Blur Levels.png', img_anom)

# Setup the plot
# Define the x range
xrange = range(len(count_results[0]))

# Setup plot labels
plt.xlabel('frame')
plt.ylabel('amount of people counted')
plt.title('Amount of people in for various degrees of anonymization')

# Plot the results
plt.plot(xrange, count_results[0], 'm', label='non-anonimized')
plt.plot(xrange, count_results[1], 'b', label='sigma 2')
plt.plot(xrange, count_results[2], 'g',  label='sigma 4')
plt.plot(xrange, count_results[3], 'r', label='sigma 6')
plt.plot(xrange, count_results[4], 'k',  label='sigma 8')
plt.legend(loc="lower right")

# Save the resuling plot
plt.savefig('Anom_comparison_result.png')
plt.close()

# Plot the average amount of people counten per amount blurring
plt.xlabel('sigma')
plt.ylabel('average amount of people counted')
plt.title('Avarege amount of people in for various degrees of anonymization')
ave_counts=[]
for list in count_results:
    ave_counts.append(sum(list)/len(list))
xrange = [0,2,4,6,8]
plt.scatter(xrange, ave_counts)
plt.scatter(xrange, ave_counts)
plt.savefig('ave_results.png')
