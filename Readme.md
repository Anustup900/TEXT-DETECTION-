# Write up 01 - Scene Text detection - ICDAR 2015 
In the present status we have implemented some SOTA models which came out of extensive reserach using the provided reserach papers and links cited below .The AI architecture flow is designed and immplemented for the indivisual models which is leading to a detection and localisation accuracy for the scene text detection on ICDAR 2015 . The model test accuracy we finally obtained on using Deep EAST with Keras OCR is 92.5 % after validation as well.

## Referene Materials : 

Apart from the reserach papers provided from the Researcher side , these are some external resources that we strongly followed and refrenced for the work : 

[1]Extensive Model Research - https://docs.google.com/document/d/1Ir0FWQI5nhnU0O92H2FTEpNWoumfUn7oZBZs8SQYcEI/edit?usp=sharing
[2]Statistical Model analysis - https://github.com/hwalsuklee/awesome-deep-text-detection-recognition
[3]List of OCR approaches with SOTA accuracy - https://github.com/zacharywhitley/awesome-ocr
[4]Scene text detection based CVPF papers - https://github.com/Jyouhou/SceneTextPapers

## Approaches and the Flow : 

As per the flow diagram provided in the repository , our algorithm is running in the following format : 
```
1.) Inputting the video to the code pipeline
2.) Video gets converted into frames
3.) Immediately the frames are passed to De convolution networks to remove blurriness and haziness from the image , simply after de convolution simply it is passed RGB contrast map using K Means distribution technique where it provided a cluster of high and low contrast images , which are deleted ( how each model works links provided below)
4.) after the data processing the frames are passed to the text localization techniques , here we had cross implemented three models ,Deep EAST ,Pixel Link & CRAFT , where EAST shown higher SOTA accuracy than the others ( How each model works - write up link provided below) 
5.) after text localization simply the coordinates are extracted through Blotzman Energy distribution technique ( support link provided below)
6.) After extraction it is passed to text recognitions models , here we hade taken 5 models for cross analysis , in which Keras OCR turned out to be the best as per the SOTA accuracy .
7.) Fianlly out puts generated ,data written over the video 
```
[1]De Convolution : https://levelup.gitconnected.com/de-blurring-images-using-convolutional-neural-networks-with-code-51d3f8d7b1d7
[2]K means for image segmentation - https://www.researchgate.net/publication/283185016_Image_Segmentation_Using_K_-means_Clustering_Algorithm_and_Subtractive_Clustering_Algorithm
[3]Deep East - https://towardsdatascience.com/neural-networks-intuitions-6-east-5892f85a097
[4]Pixel Link - https://arxiv.org/abs/1801.01315
[5]CRAFT - https://arxiv.org/abs/1904.01941
[6]Boltzman Technique - https://www.mdpi.com/2078-2489/8/4/142/pdf & https://www.researchgate.net/figure/Structure-of-restricted-Boltzmann-machine_fig5_323714457
[7] Keras - Ocr - https://core.ac.uk/download/pdf/84832286.pdf
[8]YOLO V5 - https://github.com/ultralytics/yolov5
[9]Resnet 50 - https://www.mathworks.com/help/nnet/ref/resnet50.html#:~:text=ResNet%2D50%20is%20a%20convolutional,%2C%20pencil%2C%20and%20many%20animals.
[10]R2CNN - https://arxiv.org/abs/1706.09579
[11]MObile Net - https://heartbeat.fritz.ai/real-time-object-detection-using-ssd-mobilenet-v2-on-video-streams-3bfc1577399c

