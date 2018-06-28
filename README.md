# Complementary_Segmentation_Network (Outperforms u-nets everytime :) )[under construction] 

Network Architecture for the MICCAI_2018 paper : CompNet: Complementary Segmentation Network for Brain MRI Extraction. To view the paper on Archive click the following https://arxiv.org/abs/1804.00521 Visit My website - .... to be updated shortly for intuition, hints etc



## Built With/Things Needed to implement experiments

* [Python](https://www.python.org/downloads/) - Python-2 
* [Keras](http://www.keras.io) - Deep Learning Framework used
* [Numpy](http://www.numpy.org/) - Numpy
* [Sklearn](http://scikit-learn.org/stable/install.html) - Scipy/Sklearn/Scikit-learn
* [CUDA](https://developer.nvidia.com/cuda-80-ga2-download-archive) - CUDA-8
* [CUDNN](https://developer.nvidia.com/rdp/assets/cudnn_library-pdf-5prod) - CUDNN-5 You have to register to get access to CUDNN
* [OASIS](https://www.oasis-brains.org/) - Oasis-dataset website
* [12 gb TitanX]- To implement this exact network

## Basic Idea
### Pre-requisites
This architecture can be understood after learning about the U-Net [https://arxiv.org/abs/1505.04597] {PLEASE READ U-NET before reading this paper} and W-Net [https://arxiv.org/abs/1711.08506] {Optional}.
### Comp Net summary
*ROI and CO branches - 
We take the downsampling branch of a U-Net as it is, however we split the upsampling branch into two halves, one to obtain the Region of Interest and the other for Complementary aka non region of interest. Losses here are negative dice for ROI and positive dice for Non-ROI region.*

*Reconstruction Branch - 
Next we merge these two ROI and non ROI outputs using "Summation" operation and then pass it into another U-Net, This U-Net is the reconstruction branch. The input is the summed image from previous step and the output is the "original" image that we start with. The loss of reconstruction branch is MSE.*
![alt text](https://github.com/raun1/Complementary_Segmentation_Network/blob/master/fig/network.PNG)
```
The code in this repository provides only the stand alone code for this architecture. You may implement it as is, or convert it into modular structure
if you so wish. The dataset of OASIS can obtained from the link above and the preprocessiong steps involved are mentioned in the paper. 
You have to provide the inputs.
```


![alt text](https://github.com/raun1/Complementary_Segmentation_Network/blob/master/fig/sample_results.PNG)
Sample results

