from performance_checker import PerformanceChecker
from CompareDegreesOfAnonymization import compare_degrees_of_blur

show_example = 1

if show_example == 1:
	models = [	
				'yolov8x.pt',
				'yolov9e.pt',
				'runs/detect/yolov8x_Shanghai3000/weights/YOLOx_Shanghai3000.pt',
				'runs/detect/yolov8x_shanghai3000-hybrid1500_blur5/weights/yolov8x_shanghai3000-hybrid1500_blur5.pt',
				'runs/detect/v8x-jsu-3000-blur1-1500/weights/best.pt',
				'runs/detect/v8x-jsu-3000-blur5-1500/weights/v8x-jsu-3000-blur5-1500.pt',
				'runs/detect/yolov8x_jsu3000-blur3-15002/weights/v8x-jsu-3000-blur3-1500.pt'
			]

	model_labels = ['pretrained-8x', 'pretrained-9e', 'Sh', 'Sh-Hy-Blur5', 'jsu-3000-blur1-1500', 'jsu-3000-blur5-1500', 'jsu3000-blur3-15002']

	blur_levels = [1,2,3,4,5]
	dataset_path = "datasets/Yolo8_Shanghai/Yolo8_Shanghai.yaml"
	output_dir = "Report/BlurMatrix"

	PerformanceChecker.get_blur_matrix(models, dataset_path, output_dir, blur_levels, model_labels, use_split="test")

if show_example == 2:
	data_path = "./data/madras-poc.mp4"
	model_path = './YoloModels/PreTrainedModels/yolov8x-pose-p6.pt'
	sigma_values = [2,4,6,8] # represent the degrees of blur
	compare_degrees_of_blur(data_path, model_path, sigma_values)