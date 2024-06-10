from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def compare_imgsz(vid_path, model_path):
	''' This script is for analyzing the effect or changing the imgsz parameter during inference on the processing time and accuracy 
		(i.e the effect of image compression)
	
	Parameters
	----------
		vid_path = path to video file to be used as test data
		model_path = path to model to be used for detection
	'''

	# Initialize model
	model = YOLO(model_path)

	# Initialize parameters
	count_results = []
	processing_times = []
	data_path = vid_path


	# Count the people in the each frame for non-compressed
	# Open the video and retreive video parameters
	video = cv2.VideoCapture(data_path)
	assert video.isOpened(), "Error reading video file"
	w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

	# Choose amounts of compressions
	step = int((w)/6)
	start = int(w/2)
	stop = int(w) + int((w)/6)
	imgz_values = range(start, stop, step)
	#imgz_h_values = range((h/2),int(2*h+h/2),int((h/2)/10))
	print(imgz_values, len(imgz_values))

	# Analyze the video for all chosen compression values
	for i in range(len(imgz_values)):

		start_time = time.time()

		# Open the video and retrieve video parameters
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
			
			
			# Perform inteference on a compressed frame
			results = model.track(frame, persist=True,
								imgsz=imgz_values[i], classes = 0)
			
			# Extract object ID's and count people in the frame
			while True:
				try:
					track_ids = results[0].boxes.id.int().cpu().tolist()
					count = len(track_ids)
					counts.append(count)
					break
				except AttributeError:
					count = len(results[0].boxes)
					break
			
			
		# Save the results of counting in the video
		count_results.append(counts)

		# Release and close the video
		video.release()
		cv2.destroyAllWindows()

		# Calculate and the record the proccessing time
		processing_time = time.time() - start_time
		processing_times.append(processing_time)


	# Define the x range
	xrange = range(len(count_results[0]))

	# Setup plot labels
	plt.xlabel('frame')
	plt.ylabel('amount of people counted')
	plt.title('amount of people in for various degrees imgz')

	# Plot the results
	for i in range(len(imgz_values)):
		plt.plot(xrange, count_results[i], label=str(imgz_values[i]))
	plt.legend(loc="lower right")
	#plt.ylim(50,200)

	# Save the resuling plot
	plt.savefig('imgz comparison.png')
	plt.close()

	# Plot the corresponding processing times
	# Setup plot labels
	plt.xlabel('imgz')
	plt.ylabel('processing time')
	plt.title('processing time for various amount of imgz')

	# Plot the results
	print(imgz_values,processing_times)
	plt.plot(imgz_values,processing_times)
	#plt.legend(loc='upper left')

	# Save the resuling plot
	plt.savefig('imgz times.png')
	plt.close()

	# Calculate the average amount of people per frame in the video
	ave_counts=[]
	for list in count_results:
		ave_counts.append(sum(list)/len(list))


	# Plot the average counts against the prosessing time
	plt.ylabel('average amount of people counted')
	plt.xlabel('processing time')
	plt.title('The average amount of people counted for different amounts of compression')
	plt.plot(processing_times, ave_counts)
	plt.savefig('imgz variation time vs ave counts.png')
