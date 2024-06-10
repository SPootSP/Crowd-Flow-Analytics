import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *
from tqdm import tqdm

def get_frame(vid_path, frame_path):
	''' Save the first frame of the video as a single image

	Parameters
	----------
	vid_path : path to the video
	frame_path : path where the frame should be stored to
	'''
	vid = cv2.VideoCapture(vid_path)

	_, frame = vid.read()
	cv2.imwrite(frame_path, frame)
	plt.imsave(frame_path, frame)