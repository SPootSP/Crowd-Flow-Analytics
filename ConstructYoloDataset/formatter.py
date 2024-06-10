import os
import shutil
from PIL import Image
import numpy as np
import yaml
import copy

class YoloToYoloFormatter:
	
	@staticmethod
	def reformat(input_yaml_dir: str, output_yaml_dir: str, output_root_dir: str, conversion_dict: dict):
		'''
		copies and rearranges a yolo formatted dataset
		
		input_yaml_dir = the directory of the yaml file of the dataset
		output_yaml_dir = the directory of the yaml file of the dataset
		ouput_root_dir = the (relative to the yaml file or absolute) root directory of the output dataset
		conversion_dict = the keys are one of train/val/test and the values are the relative paths from the root_dir to the train/val/test dirs
							they can be mixed to move the data around to change it to and from train/val/test dataset
							example: {"test": "train"} means that the test images becomes the train images
		'''
		with open(input_yaml_dir, "r") as file:
			input_yaml = yaml.safe_load(file.read())

		input_base_dir = input_yaml["path"]
		if not os.path.isabs(input_base_dir):
			input_base_dir = os.path.abspath(os.path.join(os.path.dirname(input_yaml_dir), input_base_dir))
			print(input_base_dir)

		output_yaml = copy.deepcopy(input_yaml)
		output_yaml["path"] = output_root_dir
		del output_yaml["train"]
		del output_yaml["val"]
		del output_yaml["test"]

		for key, value in conversion_dict.items():
			output_yaml[value] = os.path.join("images", value)

			os.makedirs(os.path.join(output_root_dir, "images", value), exist_ok=True)
			os.makedirs(os.path.join(output_root_dir, "labels", value), exist_ok=True)

			# copy the data over
			shutil.copytree(os.path.join(input_base_dir, "images", value), os.path.join(output_root_dir, "images", value), dirs_exist_ok=True)
			shutil.copytree(os.path.join(input_base_dir, "labels", value), os.path.join(output_root_dir, "labels", value), dirs_exist_ok=True)

		with open(output_yaml_dir, "w") as file:
			file.write(yaml.dump(output_yaml))


	@staticmethod
	def mod_images(base_dir: str, splits: list, mod_func):
		'''
		Modifies and overwrites the images according to the function given

		base_dir = the base directory of the dataset
		splits = list of "train", "val", "test" to signify which data splits should be affected
		mod_func = function which takes and PIL image as input and returns a PIL image as output
		'''
		for split in splits:
			split_dir = os.path.join(base_dir, "images")

			for img_file in os.listdir(split_dir):
				img_dir = os.path.join(split_dir, img_file)
				
				im = Image.open(img_dir)
				im = mod_func(im)
				im.save(img_dir)