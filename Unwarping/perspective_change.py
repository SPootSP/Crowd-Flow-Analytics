import cv2 as cv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import *
from tqdm import tqdm
import pandas as pd
from ultralytics import YOLO

def line_intersection(line1, line2):
	''' Computes the intersection coordinate of two lines in 2d space

	Parameters
	----------
		line1 (a tuple of points (x, y)) : a straight line as represented by two coordinates
		line2 (a tuple of points (x, y)) : a straight line as represented by two coordinates
	
	Returns
	-------
		(x, y) : The intersection point of line1 and line2
	'''

	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return [x, y]

def point_to_perspective(M, point):
	''' Changes the 'point' from normal image perspective to the warped perspective as given by 'M'

	Parameters
	----------
		M (numpy matrix of shape (3, 3)) : Is the matrix that converts a coordinate from image perspective to a warped perspective
		point (x, y) : a point in the unwarped image

	Returns
	-------
		(x, y) : a warped point in the warped perspective
	'''
	p = point
	x = int((M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2])))
	y = int((M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2])))
	return (x, y)

def point_to_tiles(per_annot_points, tile_coords, point):
	''' Returns the 'point' as represented in the tile space coordinate system.
	assumes tiles are square after perspective change

	Parameters
	----------
		per_annot_points (list[list[2]]) : list of points (x,y coordinates). 
			Each point is an annotated point in the warped perspective. 
			In this warped perspective the tiles should be square if the annotations are done correctly. 
			This thus also means that the warped annotated points also create a rectangle.
		tile_coords (list[list[2]]) : list of points (x, y coordinates).
			Each point corresponds to the point at the same index in the per_annot_points list.
			Each point is the coordinate in tile space. (A tile is just a square on the ground that you walk on.)
		point (x, y): coordinate in warped perspective you desire the tile space coordinate of.

	Returns
	-------
		(x, y) : a point in the tile coordinate space
	'''
	x = (point[0] - per_annot_points[0][0]) / (per_annot_points[3][0] - per_annot_points[0][0]) * (tile_coords[3][0] - tile_coords[0][0])
	y = (point[0] - per_annot_points[0][0]) / (per_annot_points[3][0] - per_annot_points[0][0]) * (tile_coords[3][0] - tile_coords[0][0])
	return (x, y)

def warp_perspective(vid_path, annot_path, tile_coords, result_path, model="yolov8x-pose-p6.pt"):
		''' Runs a yolo model detection over the given video and then converts all
		Parameters
		----------
			vid_path : path to the video you want to use for detection and warping
			annot_path : path to the file with the annotations of the video.
				These annotations can be made by the annotate_frame.py file.
				It assumes that the annotation points are done in the following order:
					start with the top left corner, then the top right, then the bottom right and then bottom left.
			tile_coords : a list of points (x, y) that give the coordinates of the annotated points in tile space. It also assumes the order of:
					start with the top left corner, then the top right, then the bottom right and then bottom left.
			result_path : the path that the resulting video is written to. It will write to .mp4 format.
			model : the model weights that will be used for detection. Only accepts Yolo model weights.
		'''
		model = YOLO(model)

		vid = cv2.VideoCapture(vid_path)
		fps = vid.get(cv2.CAP_PROP_FPS)
		frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
		cols = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
		rows = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

		annot = pd.read_csv(annot_path, header=None)
		annot.columns = ["x", "y"]
		annot_points = [
			[annot.loc[0, "x"], annot.loc[0, "y"]], # top left
			[annot.loc[1, "x"], annot.loc[1, "y"]], # top right
			[annot.loc[3, "x"], annot.loc[3, "y"]], # bottom left
			[annot.loc[2, "x"], annot.loc[2, "y"]], # bottom right
		]

		# each coordinate corresponds to its annotation point, but then in tile space
		tile_coords = tile_coords

		# expand the points as much as possible such that the whole image is inside of the warped image
		# when an image warps and the same width and height is used for the warped image then it might happen
		# that parts of the warped image is cutoff. To combat this, it is computed what the new width and height must be.
		mid_point = line_intersection([annot_points[0], annot_points[3]], [annot_points[1], annot_points[2]])

		# top left expansion factor
		expansion_point_x = line_intersection([[0, 0], [0, rows]], [annot_points[0], mid_point])
		expansion_point_y = line_intersection([[0, 0], [cols, 0]], [annot_points[0], mid_point])

		tl_ef = 0.0
		if expansion_point_x[0] > expansion_point_y[0]:
			tl_ef = (mid_point[0] - expansion_point_x[0])/(mid_point[0] - annot_points[0][0])
		else:
			tl_ef = (mid_point[0] - expansion_point_y[0])/(mid_point[0] - annot_points[0][0])

		# top right expansion factor
		expansion_point_x = line_intersection([[cols, 0], [cols, rows]], [annot_points[1], mid_point])
		expansion_point_y = line_intersection([[0, 0], [cols, 0]], [annot_points[1], mid_point])

		tr_ef = 0.0
		if expansion_point_x[0] < expansion_point_y[0]:
			tr_ef = (mid_point[0] - expansion_point_x[0])/(mid_point[0] - annot_points[1][0])
		else:
			tr_ef = (mid_point[0] - expansion_point_y[0])/(mid_point[0] - annot_points[1][0])

		# bottom left expansion factor
		expansion_point_x = line_intersection([[0, 0], [0, rows]], [annot_points[2], mid_point])
		expansion_point_y = line_intersection([[0, rows], [cols, rows]], [annot_points[2], mid_point])

		bl_ef = 0.0
		if expansion_point_x[0] < expansion_point_y[0]:
			bl_ef = (mid_point[0] - expansion_point_x[0])/(mid_point[0] - annot_points[2][0])
		else:
			bl_ef = (mid_point[0] - expansion_point_y[0])/(mid_point[0] - annot_points[2][0])

		# bottom right expansion factor
		expansion_point_x = line_intersection([[cols, 0], [cols, rows]], [annot_points[3], mid_point])
		expansion_point_y = line_intersection([[0, rows], [cols, rows]], [annot_points[3], mid_point])

		br_ef = 0.0
		if expansion_point_x[0] < expansion_point_y[0]:
			br_ef = (mid_point[0] - expansion_point_x[0])/(mid_point[0] - annot_points[3][0])
		else:
			br_ef = (mid_point[0] - expansion_point_y[0])/(mid_point[0] - annot_points[3][0])
			
		expansion_factor = max([tl_ef, tr_ef, bl_ef, br_ef])

		# expand the points
		for i in range(4):
			annot_points[i][0] = expansion_factor * (annot_points[i][0] - mid_point[0]) + mid_point[0]
			annot_points[i][1] = expansion_factor * (annot_points[i][1] - mid_point[1]) + mid_point[1]

		# create perspective matrix and warp the image
		pts1 = np.float32(annot_points)
		pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

		M = cv.getPerspectiveTransform(pts1,pts2)

		# get the perspective ecuivalent of the annot points
		per_annot_points = [
			point_to_perspective(M, annot_points[i]) for i in range(4)
		]

		frames = []
		bar = tqdm(total=frame_count)
		while True:
			i = len(frames)
			ret, frame = vid.read()
			if not ret or i == 100:
				break

			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			rows,cols,ch = img.shape

			results = model.track(img, persist=True, imgsz=1280)
			boxes = results[0].boxes.xywh.cpu()

			dst = cv.warpPerspective(img,M,(rows,cols))

			for box in boxes:
				x, y, w, h = box

				# transform point to perspective
				x, y = point_to_perspective(M, (x, y+h/2))

				cv2.circle(dst, [x, y], radius=10, color=(0, 0, 255), thickness=-1)

				# transform point to tile space
				tile_coord = point_to_tiles(per_annot_points, tile_coords, (x, y))

			frames.append(dst)
			bar.update()

		animation = VideoClip(lambda t: frames[int(t*fps)], duration=len(frames)/fps)
		animation.write_videofile(result_path, codec='mpeg4', fps=fps)
		animation.close()