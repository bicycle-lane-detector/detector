# Bicycle Lane Detector

This repository contains the CNN code to segment bicycle-friendly paths in urban environments from satellite imagery. 

This project was undertaken as a two-person research project during our Bachelor's at DBHW Karlsruhe. 
The report (in German) can be found [here](https://github.com/bicycle-lane-detector/thesis/releases/download/v1.2.0/20230517-Studienarbeit_v1_2.pdf).

The U-Net-like segmentation network was trained on satellite images of cities in Lower Saxony, Germany (e.g. Hanover, Osnabrück, and Braunschweig).
The labels were automatically annotated using OpenStreetView data of bicyle lanes. However, this approach proved as an imperfect approach 
since the labels were often besides the actual path instead of directly on, which can be seen in the examples below. 
The blue segmentation is from the ground-truth labels with the red segmentation being the prediction by the network. 

![Example2](https://github.com/bicycle-lane-detector/thesis/blob/v1.2.0/Bilder/biou/best-q1-iou6156-idx903.png)
![Example1](https://github.com/bicycle-lane-detector/thesis/blob/v1.2.0/Bilder/biou/badish-q3918-iou0063-idx498.png)
![Example3](https://github.com/bicycle-lane-detector/thesis/blob/v1.2.0/Bilder/biou/medium-q5804-iou3622-idx907.png)

We chose segmentation to be able to determine on which side of the road the bicycle path is. In the future, one could try 
to use oriented bounding boxes instead as they are generally easier to learn. 

To more realistically assess the performance of the network we introduced Buffered Intersection over Union (BIoU) for raster graphics (previously used for vectors). 
