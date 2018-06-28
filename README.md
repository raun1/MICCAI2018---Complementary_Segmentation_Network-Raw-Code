# Complementary_Segmentation_Network (Outperforms u-nets) 

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
*ROI and CO branches
We take the downsampling branch of a U-Net as it is, however we split the upsampling branch into two halves, one to obtain the Region of Interest and the other for Complementary aka non region of interest. Losses here are negative dice for ROI and positive dice for Non-ROI region.*

*Reconstruction Branch
Next we merge these two ROI and non ROI outputs using "Summation" operation and then pass it into another U-Net, This U-Net is the reconstruction branch. The input is the summed image from previous step and the output is the "original" image that we start with. The loss of reconstruction branch is MSE.*
![alt text](https://github.com/raun1/Complementary_Segmentation_Network/blob/master/fig/network.PNG)
```
The code in this repository provides only the stand alone code for this architecture. You may implement it as is, or convert it into modular structure
if you so wish. The dataset of OASIS can obtained from the link above and the preprocessiong steps involved are mentioned in the paper. 
You have to provide the inputs.
```
## Some guidelines to run the network (Please follow the paper for exact details)
The following lines provide a very simple example of how you may train the network provided in this repository and obtain predictions.
#### We are not providing the complete implementation (Dataset/Preprocessing/Calculation of ROC/ACC/ etc etc) but just the core network architecture.
You may include validation set and callback function as demonstrated, however 
please note we never used those. In the following lines the network is referred as "finalmodel"
You may apply early stopping if desired, but for our experiments we didnt.

The following line demonstrated how to initialize the early stopping, Set patience to whichever epochs you wish for the network to continue once performance starts to decline. The network stops once patience+1 epochs have been reached without any improvement from the last stored best result. Counter is reset if a better performance is acheived at any stage of the countdown.
```
xyz=keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto') 
```

The inputs are X1_train,X2_train. Use Keras's model.fit function as follows to train
```
finalmodel.fit([X1_train,X2_train], [y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train],  
                batch_size=8, 
                nb_epoch=150,
                validation_data=([X1_validate,X2_validate],[y_validate,y_validate,y_validate,y_validate,y_validate,y_validate,y_validate,y_validate,y_validate]), 
                shuffle=True,
                 callbacks=[xyz], 
                class_weight=class_weightt)
#Once again please note we never used any calls backs or validation set in our experiments.
#I used a batch size of 8.
#Compared to batch size of 2-16, 8 gave the best result, higher batch size couldnt be supported on our GPU's.
#Please read the paper for more details (Link to be provided soon.)
```
You can obtain predicitions of the different outputs using the keras's model.predict function
####Set the index variable from 0-8 to obtain corresponding outputs with 8 being the final output
```
predict_x=finalmodel.predict([X1_test,X2_test],batch_size=8)[index]
```

## Contribution

All the code were written by the authors of this paper.
Please contact (raun- rd31879@uga.edu) for questions and queries and more details if you are having trouble with the complete implimentation. 

## Things to cite -

###### If you use Keras please cite it as follows - @misc{chollet2015keras,title={Keras},author={Chollet, Fran\c{c}ois and others},year={2015},publisher={GitHub},howpublished={\url{https://github.com/keras-team/keras }},}
###### If you use LIDC dataset please cite according to this webpage - https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#3a2715226d2841ee9ce0ba722f66232f, more details also mentioned in our paper
######

![alt text](https://github.com/raun1/Complementary_Segmentation_Network/blob/master/fig/sample_results.PNG)
Sample results

