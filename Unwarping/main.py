from perspective_change import warp_perspective
from annotate_frame import annotate_frame
from get_frame import get_frame

show_example = 1

if show_example == 1:
	# example for the perspective warping
	# please adjust to your specific case
	vid_path = "./2023-madras-poc-dataset/video/h_movie4K_A_BASELINE.mp4"
	annot_path = "./src/Perspective/annotations.csv"
	result_path = "./src/Perspective/results.mp4"

	tile_coords = [
		[0, 0],
		[20, 0],
		[0, 22],
		[20, 22]	#(x = 20 tiles and y = 22 tiles (as in stone tiles on the floor))
	]

	# tile coordinates are used as a ground truth to represent x and y coordinates on a 2d floor map
	# warp_perspective will adjust the model detections from the image and converts them a 2d floor map coordinate system
	# which is in this case the tiles on the floor
	warp_perspective(vid_path, annot_path, tile_coords, result_path)

if show_example == 2:
	# parameters
	# please adjust to your specific case
	input_path = "./frame.png"
	output_path = "./annotations.csv"
	annotate_frame(input_path, output_path)

if show_example == 3:
	# please adjust to your specific case
	vid_path = "./2023-madras-poc-dataset/video/h_movie4K_A_BASELINE.mp4"
	result_path = "./src/Perspective/frame.png"

	get_frame(vid_path, result_path)