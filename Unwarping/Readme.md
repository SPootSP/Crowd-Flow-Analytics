## Purpose
When a model returns its detections in a video or image it will always return the coordinates as x and y pixel coordinates.
But when you have multiple camera setups each pointing at different places at different angles it is usefull to combine the detections
of all these camera's. Thus the idea is to convert the detections to a unified coordinate system of a floormapping. In the example, the floormapping is according to the tiles on the floor of the video.

## Files
 - main.py : an example file to express how to use the other files
 - get_frame.py : contains a function that saves a single frame from a video
 - annotate_frame.py : contains the annotate_frame function which helps in choosing the right coordinates of the floor tiles. Be carefull of the order you choose the annotation points since this matter a lot but comments will help you with that. Also make sure that the box you create is perfectly aligned to the tiles on the floor in the frame.
 - perspective_change.py : runs a detection model over the video and then converts the detection coordinates to the unified coordinate system of a floormapping.