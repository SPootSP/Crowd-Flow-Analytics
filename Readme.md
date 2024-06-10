# Report

## Folders
 - BlurMatrix
 - ConstructYoloDataset
 - ProcessingTimeAnalysis
 - Training&Validation
 - Unwarping
 
 - YoloModels: this folder contains the variouse weights of the Yolo model that are used for the report. The weights correspond to training on different datasets and thus are labelled with the dataset name. Model weights with 'Blurx' in them. the x value means the amount of blur used on the training images. The x represents the sigma value of Gaussian blur with (0, 0) stride (python: `cv2.(img, (0,0), x)`). In case there is only a single number such as 5000, then it means trained on a non blurred dataset with 5000 epochs. When 'base' is in the name it is also trained no non blurred dataset but we don't know the amount of epochs anymore.