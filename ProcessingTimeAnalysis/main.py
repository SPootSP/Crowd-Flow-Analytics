from FrameRateComparison import compare_framerates
from ImgszComparison import compare_imgsz

show_example = 1

if show_example == 1:
	frame_skips = [2,3,5,10]
	data_path = "./data/madras-poc.mp4"
	model_path = './YoloModels/PreTrainedModels/yolov8x.pt'
	compare_framerates(frame_skips, vid_path=data_path, model_path=model_path)

if show_example == 2:
	data_path = "./data/madras-poc.mp4"
	model_path = "YoloModels/PreTrainedModels/yolov8x-pose-p6.pt"
	compare_imgsz(model_path=model_path, vid_path=data_path)