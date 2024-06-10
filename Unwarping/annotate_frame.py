import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import pandas as pd

def annotate_frame(input_path, output_path):
	''' Given a frame from a video. it will help the user select the appropriate coordinates as annotations
	These annotations then can later be used to warp the video such that the ground is square, which itself
	can be used to represent detections as a 2d coordinate of a floormap.

	Parameters
	----------
		input_path : a string that gives the path to a single frame of the video
		output_path : a string path to a .csv file that will contain the annotation points
	'''
	# variables
	points = pd.DataFrame(columns=["x", "y"])
	img = plt.imread(input_path)

	def redraw():
		global points, img

		plt.imshow(img)
		plt.plot(points["x"], points["y"])

		for i in range(points.shape[0]-1):
			plt.axline((points.loc[i, "x"], points.loc[i, "y"]), (points.loc[i+1, "x"], points.loc[i+1, "y"]))

		if len(points) == 4:
			plt.axline((points.loc[3, "x"], points.loc[3, "y"]), (points.loc[0, "x"], points.loc[0, "y"]))

		plt.draw()

	def on_click(event):
		global points
		if event.button is MouseButton.LEFT:
			if points.shape[0] < 4:
				points.loc[len(points)] = [event.xdata, event.ydata]
				redraw()
			
		if event.button is MouseButton.RIGHT:
			points = pd.DataFrame(columns=["x", "y"])
			redraw()
			

	print("Click make a square that is alligned with the ground, start with the top left corner, then the top right, then the bottom right and so on.")
	print("Click with the right mouse button to cancel the selection")

	plt.connect('button_release_event', on_click)

	redraw()
	plt.show()

	points.to_csv(output_path, header=None, index=False)