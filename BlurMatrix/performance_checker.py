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
from pprint import pprint

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
	def get_blur_matrix(models, dataset_dir, output_dir, blur_levels, model_labels = None, use_split="val"):
		'''
		This script is for comparing the performance of various models on different levels of blurring. 
		List your model directories and the blurring you want to evaluate with.

		Parameters
		----------
			models 			: is a list of directories to the weigths of the models you want to check
			dataset_dir 	: is a path to the yaml file of the dataset you want to use as validation, it will automatically pick the validation split of the dataset
			output_dir		: the directory where the blur matrices are stored
			blur_levels 	: is a list of sigmas used for blurring the dataset
			model_labels 	: is an optional argument that is a list of labels for each of the models that is used on the model axis of the blur matrix
			use_split 		: specifies what split of the dataset to use for validation
		'''

		"fitness" "precision, recall, mAP50, mAP50-95, f1"

		# Initialize parameters
		validation_img_dir = YoloToYoloFormatter.get_img_dir(dataset_dir, use_split)
		validation_label_dir = YoloToYoloFormatter.get_label_dir(dataset_dir, use_split)

		files = [os.path.splitext(os.path.basename(file))[0] for file in os.listdir(validation_img_dir)]
		img_files = {os.path.splitext(os.path.basename(file))[0]: os.path.join(validation_img_dir, file) for file in os.listdir(validation_img_dir)}
		label_files = {os.path.splitext(os.path.basename(file))[0]: os.path.join(validation_label_dir, file) for file in os.listdir(validation_label_dir)}

		metrics = ["precision", "recall", "mAP50", "mAP50_95", "f1", "fitness"]
		counts = {key: np.zeros((len(models), len(blur_levels))) for key in metrics}
		for j in range(len(blur_levels)):
			number = random.randint(0, sys.maxsize * 2 + 1)
			os.makedirs(f"/tmp/{number}", exist_ok=True)

			if use_split == "val":
				YoloToYoloFormatter.copy(dataset_dir, f"/tmp/{number}/temp.yaml", f"/tmp/{number}/", conversion_dict={"val": "val", "train": "train"})
			elif use_split == "test":
				YoloToYoloFormatter.copy(dataset_dir, f"/tmp/{number}/temp.yaml", f"/tmp/{number}/", conversion_dict={"test": "val", "train": "train"})
			else:
				YoloToYoloFormatter.copy(dataset_dir, f"/tmp/{number}/temp.yaml", f"/tmp/{number}/", conversion_dict={"train": "val", "train": "train"})
			
			YoloToYoloFormatter.mod_images(dataset_dir, [use_split, "train"], lambda img: PerformanceChecker.__blur_img(img, blur_levels[j]))

			for i in range(len(models)):
				print(f"progess: {i + j * len(models)}/{len(blur_levels) * len(models)}, model: {models[i]}, blur level: {blur_levels[j]}")

				# Choose the model
				model = YOLO(models[i])

				results = model.val(data=f"/tmp/{number}/temp.yaml", device="cuda")

				try:
					precision = results.box.class_result(0)[0]
					recall = results.box.class_result(0)[1]
					mAP50 = results.box.class_result(0)[2]
					mAP50_95 = results.box.class_result(0)[3]
					f1 = results.box.f1[0]
					fitness = results.box.fitness()

					counts["precision"][i, j] = precision
					counts["recall"][i, j] = recall
					counts["mAP50"][i, j] = mAP50
					counts["mAP50_95"][i, j] = mAP50_95
					counts["f1"][i, j] = f1
					counts["fitness"][i, j] = fitness

				except IndexError as e:
					counts["precision"][i, j] = 0
					counts["recall"][i, j] = 0
					counts["mAP50"][i, j] = 0
					counts["mAP50_95"][i, j] = 0
					counts["f1"][i, j] = 0
					counts["fitness"][i, j] = 0

			shutil.rmtree(f"/tmp/{number}/")

		# Set up the plot
		for key in metrics:
			plt.title(f'Models used on different blurs with metric {key}')

			plt.matshow(counts[key])
			plt.colorbar()
			plt.xticks(range(len(blur_levels)), blur_levels)
			plt.yticks(range(len(models)), model_labels)

			plt.savefig(os.path.join(output_dir, f'matrix plot models vs blur - metric {key}'))
			plt.close()