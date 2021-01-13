# Nuclear Detection and Classification
This is the code for nuclear detection and classification on digitalized H&E whole slide images.

## How to use? 
Just open the file __DET_REG_NET.py__ and it contains many functions/modules that you may want.

## What we do?
We initialize this project from [paper](https://www.tandfonline.com/doi/pdf/10.1080/21681163.2016.1149104?needAccess=true), and the DET_REG_NET 
is the implementation of FCRN-A. We further make the encoder deeper and also try to use UNet structure to polish the upsampling results.

## TODO
- Only use MSE loss since it is quite a easy problem. Maybe we could use Smooth L1 for paper :).  
- We did not consider the overfit and just use one image patch(ICC cancer) to test its peformance.  
- We left the Conv2d as the last layer, i am not sure if it is right(I do normlization before postprocessing every time).

## Notice 
You may notice the nms module in file, but its not well implemented so be careful to use it. 

## Thanks
The idea of this code is originates from [paper](https://www.tandfonline.com/doi/pdf/10.1080/21681163.2016.1149104?needAccess=true).
