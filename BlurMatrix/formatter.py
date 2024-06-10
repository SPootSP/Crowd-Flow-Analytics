import os
import shutil
from PIL import Image
import numpy as np
import yaml as yml
import copy
import tqdm

class YoloToYoloFormatter:
	
	@staticmethod
	def copy(input_yaml_dir: str, output_yaml_dir: str, output_root_dir: str, conversion_dict: dict):
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
			input_yaml = yml.safe_load(file.read())

		for key in conversion_dict.keys():
			assert key in input_yaml.keys() and input_yaml[key] is not None, f"The input dataset does not have a {key} split."

		input_base_dir = input_yaml["path"]
		if not os.path.isabs(input_base_dir):
			input_base_dir = os.path.abspath(os.path.join(os.path.dirname(input_yaml_dir), input_base_dir))

		output_yaml = copy.deepcopy(input_yaml)
		output_yaml["path"] = output_root_dir
		del output_yaml["train"]
		del output_yaml["val"]
		del output_yaml["test"]

		for key, value in conversion_dict.items():
			output_yaml[value] = os.path.join("images", value)

			with open(output_yaml_dir, "w") as file:
				file.write(yml.dump(output_yaml))

			os.makedirs(YoloToYoloFormatter.get_img_dir(output_yaml_dir, value), exist_ok=True)
			os.makedirs(YoloToYoloFormatter.get_label_dir(output_yaml_dir, value), exist_ok=True)

			# copy the data over
			shutil.copytree(YoloToYoloFormatter.get_img_dir(input_yaml_dir, key), YoloToYoloFormatter.get_img_dir(output_yaml_dir, value), dirs_exist_ok=True)
			shutil.copytree(YoloToYoloFormatter.get_label_dir(input_yaml_dir, key), YoloToYoloFormatter.get_label_dir(output_yaml_dir, value), dirs_exist_ok=True)

		with open(output_yaml_dir, "w") as file:
			file.write(yml.dump(output_yaml))


	@staticmethod
	def mod_images(yaml_dir: str | dict, splits: list, mod_func):
		'''
		Modifies and overwrites the images according to the function given

		Parameters
		----------
			yaml_dir : directory of the yaml file of the dataset
			splits : list of "train", "val", "test" to signify which data splits should be affected
			mod_func : function which takes and PIL image as input and returns a PIL image as output
		'''
		for split in splits:
			split_dir = YoloToYoloFormatter.get_img_dir(yaml_dir, split)

			print("Modifying the images, can take a while")
			for img_file in tqdm.tqdm(os.listdir(split_dir)):
				img_dir = os.path.join(split_dir, img_file)
				
				im = Image.open(img_dir)
				im = mod_func(im)
				im.save(img_dir)

	@staticmethod
	def get_img_dir(yaml_dir: str | dict, split: str):
		'''
		returns the directory of where the validation images are stored

		Parameters
		----------
			yaml_dir : directory of the yaml file of the dataset
			split : one of "train"/"val"/"test"
		'''

		with open(yaml_dir, "r") as file:
			input_yaml = yml.safe_load(file.read())

		assert split in input_yaml.keys() and input_yaml[split] is not None, f"the dataset does not contain the {split} split"

		if os.path.isabs(input_yaml["path"]):
			root_dir = os.path.join(input_yaml["path"], input_yaml[split])
		else:
			root_dir = os.path.join(os.path.dirname(yaml_dir), input_yaml["path"], input_yaml[split])

		return root_dir
	
	@staticmethod
	def get_label_dir(yaml_dir: str | dict, split: str):
		'''
		returns the directory of where the validation labels are stored

		Parameters
		----------
			yaml_dir : directory of the yaml file of the dataset
			split : one of "train"/"val"/"test"
		'''

		with open(yaml_dir, "r") as file:
			input_yaml = yml.safe_load(file.read())

		assert split in input_yaml.keys() and input_yaml[split] is not None, f"the dataset does not contain the {split} split"

		if os.path.isabs(input_yaml["path"]):
			root_dir = os.path.join(input_yaml["path"], input_yaml[split].replace("images", "labels"))
		else:
			root_dir = os.path.join(os.path.dirname(yaml_dir), input_yaml["path"], input_yaml[split].replace("images", "labels"))

		return root_dir