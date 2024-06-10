from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from torchvision.ops import box_iou
import os
from torch import tensor, from_numpy
from PIL import ImageFilter 
import random
import sys
import shutil

from formatter import YoloToYoloFormatter

class PerformanceChecker:

	@staticmethod
	def __blur_img(img, radius):
		'''
		private function

		blurs an image
		img = PIL image
		radius = radius value of the blur
		'''
		return img.filter(ImageFilter.GaussianBlur(radius=radius))

	@staticmethod
	def get_blur_matrix(models, dataset_dir, blur_levels, model_labels = None):
		'''
		This script is for comparing the performance of various models on different levels of blurring. 
		List your model directories and the blurring you want to evaluate with.

		models is a list of directories to the weigths of the models you want to check
		dataset_dir is a path to the yaml file of the dataset you want to use as validation, it will automatically pick the validation split of the dataset
		blur_levels is a list of sigmas used for blurring the dataset
		model_labels is an optional argument that is a list of labels for each of the models that is used on the model axis of the blur matrix
		'''

		# Initialize parameters
		validation_img_dir = YoloToYoloFormatter.get_img_dir(dataset_dir, "val")
		validation_label_dir = YoloToYoloFormatter.get_label_dir(dataset_dir, "val")

		files = [os.path.splitext(os.path.basename(file))[0] for file in os.listdir(validation_img_dir)]
		img_files = {os.path.splitext(os.path.basename(file))[0]: os.path.join(validation_img_dir, file) for file in os.listdir(validation_img_dir)}
		label_files = {os.path.splitext(os.path.basename(file))[0]: os.path.join(validation_label_dir, file) for file in os.listdir(validation_label_dir)}

		blur_counts = []
		for j in range(len(blur_levels)):
			number = random.randint(0, sys.maxsize * 2 + 1)
			os.makedirs(f"/tmp/{number}", exist_ok=True)
			YoloToYoloFormatter.copy(dataset_dir, f"/tmp/{number}/temp.yaml", f"/tmp/{number}/", conversion_dict={"val": "val"})
			YoloToYoloFormatter.mod_images(dataset_dir, ["val"], lambda img: PerformanceChecker.__blur_img(img, blur_levels[j]))

			model_counts = []
			for i in range(len(models)):
				print(f"progess: {i + j * len(models)}/{len(blur_levels) * len(models)}, model: {models[i]}, blur level: {blur_levels[j]}")

				# Choose the model
				model = YOLO(models[i])

				# Perform inference over the whole video for every level of blurring
				count_results = []

				# get images
				for k in range(0, len(files), 16):

					imgs = []
					labels = np.zeros((0, 4))
					for file in files[k:k+16]:
						img = cv2.imread(img_files[file])
						imgs.append(img)

						label = np.genfromtxt(label_files[file], delimiter=' ')
						if len(label.shape) == 2: # otherwise assume the file is empty
							label = label[label[:, 0] == 0]	# remove all non person labels
							labels = np.concatenate((labels, label[:, 1:]))

					results = model(imgs)

					temp = np.zeros((0, 4))
					for result in results:
						temp = np.concatenate((temp, result.boxes.xyxy.cpu()))

					target = from_numpy(labels)
					pred = from_numpy(temp)

					iou = box_iou(pred, target)

					if len(iou.shape) == 2:
						iou = np.max(iou.numpy(), axis=0)	# get the iou of the target box that best fits the pred box
					print(iou.shape, pred.shape, target.shape)
					print(np.average(iou))
					count_results.append(np.average(iou))

				model_counts.append(count_results)
				## Open the video
				#video = cv2.VideoCapture("./data/Busy Platform.mp4")
				#assert video.isOpened(), "Error reading video file"

				## Retrieve video parameters
				#w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

					
				#	# Extract the tracking ID's and count the objects in the frame and store the results  
				#	try:  
				#		boxes = results[0].boxes.xywh.cpu()
				#		bottoms = np.zeros((len(boxes), 1))

				#		for i in range(len(boxes)):
				#			_, yb, _, hb = boxes[i]
				#			bottoms[i] = yb + hb/2

				#		hist, edges = np.histogram(bottoms, bins=10)

				#		counts.append(hist)

				#	except AttributeError:
				#		counts.append([0 for i in range(len(10))])

				## Save the results of counting in the video
				#print(np.array(counts).shape)
				#count_results.append(np.array(counts))

				## Release and close the video
				#video.release()
				#cv2.destroyAllWindows()

			blur_counts.append(np.array(model_counts))
			shutil.rmtree(f"/tmp/{number}/")
		#setup the plot

		ave_array = np.average(np.array(blur_counts))
		print(ave_array.shape)
		#ave_array = np.array(model_counts).reshape(len(models)*10,len(blurs))

		# Set up the plot
		plt.title('Models used on different blurs')

		plt.matshow(ave_array)
		plt.colorbar()
		#plt.xticks(range(len(blurs)), blurs)
		#plt.yticks(range(len(models)*10), Models_label)

		plt.savefig('matrix plot models vs blur')
		plt.close()

		print(processing_times,count_results,ave_counts,ave_array)