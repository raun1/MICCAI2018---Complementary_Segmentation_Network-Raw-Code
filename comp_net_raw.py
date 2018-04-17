
# coding: utf-8

# In[2]:

import keras
import scipy as sp
import scipy.misc, scipy.ndimage.interpolation
from medpy import metric
import numpy as np
import os
from keras import losses
import tensorflow as tf
from keras.models import Model
from keras.layers import Input,merge, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D,Dropout,Conv2DTranspose,add,multiply
from keras.layers.normalization import BatchNormalization as bn
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import RMSprop
from keras import regularizers 
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np 
import nibabel as nib
CUDA_VISIBLE_DEVICES = [1]
os.environ['CUDA_VISIBLE_DEVICES']=','.join([str(x) for x in CUDA_VISIBLE_DEVICES])
#oasis files 1-457

#path='/home/bahaa/oasis_mri/OAS1_'


# In[3]:

import numpy as np
import cv2







#Dice coeff
smooth = 1.
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def neg_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (1-((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)))


def neg_dice_coef_loss(y_true, y_pred):
    return dice_coef(y_true, y_pred)


# In[7]:


#define the model

#define the model
def UNet(input_shape,learn_rate=1e-3):
    l2_lambda = 0.0002
    DropP = 0.3
    kernel_size=3

    inputs = Input(input_shape)

    conv1a = Conv2D( 12, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(inputs)
    
    
    conv1a = bn()(conv1a)
    
    conv1b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv1a)

    conv1b = bn()(conv1b)

    merge1=concatenate([conv1a,conv1b])
    
    conv1c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv1c = bn()(conv1c)

    merge2=concatenate([conv1a,conv1b,conv1c])

    conv1d = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv1d = bn()(conv1d)

    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1d)

    pool1 = Dropout(DropP)(pool1)



    

    conv2a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
    
    conv2a = bn()(conv2a)

    conv2b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv2a)

    conv2b = bn()(conv2b)

    merge1=concatenate([conv2a,conv2b])

    conv2c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv2c = bn()(conv2c)

    merge2=concatenate([conv2a,conv2b,conv2c])

    conv2d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv2d = bn()(conv2d)


    merge3=concatenate([conv2a,conv2b,conv2c,conv2d])



    conv2e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv2e = bn()(conv2e)

    merge4=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e])


    conv2f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv2f = bn()(conv2f)


    merge5=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f])

    conv2g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv2g = bn()(conv2g)

    merge6=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f,conv2g])


    conv2h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv2h = bn()(conv2h)

    merge7=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f,conv2g,conv2h])


    conv2i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv2i = bn()(conv2g)

    merge8=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f,conv2g,conv2h,conv2i])

    conv2j = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv2j = bn()(conv2g)

    
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2j)

    pool2 = Dropout(DropP)(pool2)







    conv3a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)
    
    conv3a = bn()(conv3a)

    conv3b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv3a)

    conv3b = bn()(conv3b)

    merge1=concatenate([conv3a,conv3b])

    conv3c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv3c = bn()(conv3c)

    merge2=concatenate([conv3a,conv3b,conv3c])

    conv3d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv3d = bn()(conv3d)


    merge3=concatenate([conv3a,conv3b,conv3c,conv3d])



    conv3e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv3e = bn()(conv3e)

    merge4=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e])


    conv3f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv3f = bn()(conv3f)


    merge5=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f])

    conv3g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv3g = bn()(conv3g)

    merge6=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g])


    conv3h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv3h = bn()(conv3h)

    merge7=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h])


    conv3i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv3i = bn()(conv3i)

    merge8=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i])

    conv3j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv3j = bn()(conv3j)


    merge9=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j])


    conv3k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    conv3k = bn()(conv3k)


    merge10=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k])
    conv3l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    conv3l = bn()(conv3l)

    merge11=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l])
    conv3m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    conv3m = bn()(conv3m)


    merge12=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m])
    conv3n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    conv3n = bn()(conv3n)




    merge13=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n])
    conv3o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    conv3o = bn()(conv3o)



    merge14=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o])
    conv3p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    conv3p = bn()(conv3p)


    merge15=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p])
    conv3q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    conv3q = bn()(conv3q)


    merge16=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q])
    conv3r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    conv3r = bn()(conv3r)


    merge17=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q,conv3r])
    conv3s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    conv3s = bn()(conv3s)


    merge18=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q,conv3r,conv3s])
    conv3t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    conv3t = bn()(conv3t)

    merge19=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q,conv3r,conv3s,conv3t])
    conv3u=Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    conv3u = bn()(conv3u)


    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3u)

    pool3 = Dropout(DropP)(pool3)

    
    conv4a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
    
    conv4a = bn()(conv4a)

    conv4b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv4a)

    conv4b = bn()(conv4b)

    merge1=concatenate([conv4a,conv4b])

    conv4c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv4c = bn()(conv4c)

    merge2=concatenate([conv4a,conv4b,conv4c])

    conv4d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv4d = bn()(conv4d)


    merge3=concatenate([conv4a,conv4b,conv4c,conv4d])



    conv4e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv4e = bn()(conv4e)

    merge4=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e])


    conv4f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv4f = bn()(conv4f)


    merge5=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f])

    conv4g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv4g = bn()(conv4g)

    merge6=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g])


    conv4h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv4h = bn()(conv4h)

    merge7=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h])


    conv4i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv4i = bn()(conv4i)

    merge8=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i])

    conv4j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv4j = bn()(conv4j)


    merge9=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j])


    conv4k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    conv4k = bn()(conv4k)


    merge10=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k])
    conv4l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    conv4l = bn()(conv4l)

    merge11=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l])
    conv4m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    conv4m = bn()(conv4m)


    merge12=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m])
    conv4n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    conv4n = bn()(conv4n)




    merge13=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n])
    conv4o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    conv4o = bn()(conv4o)



    merge14=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o])
    conv4p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    conv4p = bn()(conv4p)


    merge15=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p])
    conv4q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    conv4q = bn()(conv4q)


    merge16=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q])
    conv4r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    conv4r = bn()(conv4r)


    merge17=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q,conv4r])
    conv4s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    conv4s = bn()(conv4s)


    merge18=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q,conv4r,conv4s])
    conv4t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    conv4t = bn()(conv4t)

    merge19=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q,conv4r,conv4s,conv4t])
    conv4u=Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    conv4u = bn()(conv4u)
    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4u)

    pool4 = Dropout(DropP)(pool4)





    conv5a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
    
    conv5a = bn()(conv5a)

    conv5b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(conv5a)

    conv5b = bn()(conv5b)

    merge1=concatenate([conv5a,conv5b])

    conv5c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv5c = bn()(conv5c)

    merge2=concatenate([conv5a,conv5b,conv5c])

    conv5d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv5d = bn()(conv5d)


    merge3=concatenate([conv5a,conv5b,conv5c,conv5d])



    conv5e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv5e = bn()(conv5e)

    merge4=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e])


    conv5f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv5f = bn()(conv5f)


    merge5=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f])

    conv5g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv5g = bn()(conv5g)

    merge6=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g])


    conv5h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv5h = bn()(conv5h)

    merge7=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h])


    conv5i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv5i = bn()(conv5i)

    merge8=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i])

    conv5j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv5j = bn()(conv5j)


    merge9=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j])


    conv5k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    conv5k = bn()(conv5k)


    merge10=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k])
    conv5l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    conv5l = bn()(conv5l)

    merge11=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l])
    conv5m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    conv5m = bn()(conv5m)


    merge12=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m])
    conv5n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    conv5n = bn()(conv5n)




    merge13=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n])
    conv5o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    conv5o = bn()(conv5o)



    merge14=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o])
    conv5p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    conv5p = bn()(conv5p)


    merge15=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p])
    conv5q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    conv5q = bn()(conv5q)


    merge16=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q])
    conv5r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    conv5r = bn()(conv5r)


    merge17=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q,conv5r])
    conv5s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    conv5s = bn()(conv5s)


    merge18=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q,conv5r,conv5s])
    conv5t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    conv5t = bn()(conv5t)

    merge19=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q,conv5r,conv5s,conv5t])
    conv5u=Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    conv5u = bn()(conv5u)
    



    up6 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(conv5u), conv4u],name='up6', axis=3)

    out6=Conv2DTranspose(12,(2, 2), strides=(8, 8), padding='same')(up6)
    out6 = bn()(out6)
    output1 = Conv2D(1, (1, 1), activation='sigmoid',name='output1')(out6)

    up6 = Dropout(DropP)(up6)

    conv6a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up6)
    
    conv6a = bn()(conv6a)

    merge0=concatenate([up6,conv6a])

    conv6b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    conv6b = bn()(conv6b)

    merge1=concatenate([up6,conv6a,conv6b])

    conv6c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv6c = bn()(conv6c)

    merge2=concatenate([up6,conv6a,conv6b,conv6c])

    conv6d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv6d = bn()(conv6d)


    merge3=concatenate([up6,conv6a,conv6b,conv6c,conv6d])


    conv6e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv6e = bn()(conv6e)

    merge4=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e])


    conv6f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv6f = bn()(conv6f)


    merge5=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f])

    conv6g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv6g = bn()(conv6g)

    merge6=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g])


    conv6h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv6h = bn()(conv6h)

    merge7=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h])


    conv6i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv6i = bn()(conv6i)

    merge8=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i])

    conv6j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv6j = bn()(conv6j)


    merge9=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j])


    conv6k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    conv6k = bn()(conv6k)


    merge10=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k])
    conv6l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    conv6l = bn()(conv6l)

    merge11=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l])
    conv6m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    conv6m = bn()(conv6m)


    merge12=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m])
    conv6n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    conv6n = bn()(conv6n)




    merge13=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n])
    conv6o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    conv6o = bn()(conv6o)



    merge14=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n,conv6o])
    conv6p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    conv6p = bn()(conv6p)


    merge15=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n,conv6o,conv6p])
    conv6q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    conv6q = bn()(conv6q)


    merge16=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n,conv6o,conv6p,conv6q])
    conv6r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    conv6r = bn()(conv6r)


    merge17=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n,conv6o,conv6p,conv6q,conv6r])
    conv6s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    conv6s = bn()(conv6s)


    merge18=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n,conv6o,conv6p,conv6q,conv6r,conv6s])
    conv6t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    conv6t = bn()(conv6t)

    merge19=concatenate([up6,conv6a,conv6b,conv6c,conv6d,conv6e,conv6f,conv6g,conv6h,conv6i,conv6j,conv6k,conv6l,conv6m,conv6n,conv6o,conv6p,conv6q,conv6r,conv6s,conv6t])
    conv6u=Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    conv6u = bn()(conv6u)





    up7 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(conv6u), conv3u],name='up7', axis=3)

    up7 = Dropout(DropP)(up7)
    #add second output here
    out7=Conv2DTranspose(12,(2, 2), strides=(4, 4), padding='same')(up7)
    out7 = bn()(out7)
    output2 = Conv2D(1, (1, 1), activation='sigmoid',name='output2')(out7)
    conv7a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up7)
    
    conv7a = bn()(conv7a)

    merge0=concatenate([up7,conv7a])

    conv7b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    conv7b = bn()(conv7b)

    merge1=concatenate([up7,conv7a,conv7b])

    conv7c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv7c = bn()(conv7c)

    merge2=concatenate([up7,conv7a,conv7b,conv7c])

    conv7d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv7d = bn()(conv7d)


    merge3=concatenate([up7,conv7a,conv7b,conv7c,conv7d])



    conv7e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv7e = bn()(conv7e)

    merge4=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e])


    conv7f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv7f = bn()(conv7f)


    merge5=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f])

    conv7g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv7g = bn()(conv7g)

    merge6=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g])


    conv7h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv7h = bn()(conv7h)

    merge7=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h])


    conv7i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv7i = bn()(conv7i)

    merge8=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i])

    conv7j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv7j = bn()(conv7j)


    merge9=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j])


    conv7k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    conv7k = bn()(conv7k)


    merge10=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k])
    conv7l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    conv7l = bn()(conv7l)

    merge11=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l])
    conv7m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    conv7m = bn()(conv7m)


    merge12=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m])
    conv7n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    conv7n = bn()(conv7n)




    merge13=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n])
    conv7o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    conv7o = bn()(conv7o)



    merge14=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n,conv7o])
    conv7p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    conv7p = bn()(conv7p)


    merge15=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n,conv7o,conv7p])
    conv7q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    conv7q = bn()(conv7q)


    merge16=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n,conv7o,conv7p,conv7q])
    conv7r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    conv7r = bn()(conv7r)


    merge17=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n,conv7o,conv7p,conv7q,conv7r])
    conv7s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    conv7s = bn()(conv7s)


    merge18=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n,conv7o,conv7p,conv7q,conv7r,conv7s])
    conv7t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    conv7t = bn()(conv7t)

    merge19=concatenate([up7,conv7a,conv7b,conv7c,conv7d,conv7e,conv7f,conv7g,conv7h,conv7i,conv7j,conv7k,conv7l,conv7m,conv7n,conv7o,conv7p,conv7q,conv7r,conv7s,conv7t])
    conv7u=Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    conv7u = bn()(conv7u)




    up8 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(conv7u), conv2j],name='up8', axis=3)

    up8 = Dropout(DropP)(up8)
    #add third outout here
    out8=Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(up8)
    out8 = bn()(out8)
    output3 = Conv2D(1, (1, 1), activation='sigmoid',name='output3')(out8)
    conv8a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up8)
    
    conv8a = bn()(conv8a)

    merge0=concatenate([up8,conv8a])

    conv8b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    conv8b = bn()(conv8b)

    merge1=concatenate([up8,conv8a,conv8b])

    conv8c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv8c = bn()(conv8c)

    merge2=concatenate([up8,conv8a,conv8b,conv8c])

    conv8d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv8d = bn()(conv8d)


    merge3=concatenate([up8,conv8a,conv8b,conv8c,conv8d])



    conv8e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    conv8e = bn()(conv8e)

    merge4=concatenate([up8,conv8a,conv8b,conv8c,conv8d,conv8e])


    conv8f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    conv8f = bn()(conv8f)


    merge5=concatenate([up8,conv8a,conv8b,conv8c,conv8d,conv8e,conv8f])

    conv8g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    conv8g = bn()(conv8g)

    merge6=concatenate([up8,conv8a,conv8b,conv8c,conv8d,conv8e,conv8f,conv8g])


    conv8h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    conv8h = bn()(conv8h)

    merge7=concatenate([up8,conv8a,conv8b,conv8c,conv8d,conv8e,conv8f,conv8g,conv8h])


    conv8i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    conv8i = bn()(conv8i)

    merge8=concatenate([up8,conv8a,conv8b,conv8c,conv8d,conv8e,conv8f,conv8g,conv8h,conv8i])

    conv8j = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    conv8j = bn()(conv8j)


    
    up9 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(conv8j), conv1d],name='up9',axis=3)

    up9 = Dropout(DropP)(up9)
    out9=Conv2DTranspose(12,(2, 2), strides=(1, 1), padding='same')(up9)
    out9 = bn()(out9)
    output4 = Conv2D(1, (1, 1), activation='sigmoid',name='output4')(out9)

    conv9a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(up9)
    
    conv9a = bn()(conv9a)

    merge0=concatenate([up9,conv9a])

    conv9b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    conv9b = bn()(conv9b)

    merge1=concatenate([up9,conv9a,conv9b])

    conv9c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    conv9c = bn()(conv9c)

    merge2=concatenate([up9,conv9a,conv9b,conv9c])

    conv9d = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    conv9d = bn()(conv9d)

   
    conv10 = Conv2D(1, (1, 1), activation='sigmoid',name='conv10')(conv9d)

    finalmerge=concatenate([out6,out7,out8,out9,conv9d])

    final_op=Conv2D(1, (1, 1), activation='sigmoid',name='final_op')(finalmerge)

#    model = Model(inputs=inputs, outputs=[out6,out7,out8,out9,conv10,final_op])


    


    #second branch - brain
    xup6 = concatenate([Conv2DTranspose(24,(2, 2), strides=(2, 2), padding='same')(conv5u), conv4u],name='xup6', axis=3)

    xout6=Conv2DTranspose(24,(2, 2), strides=(8, 8), padding='same')(xup6)
    xout6 = bn()(xout6)
    xoutput1 = Conv2D(1, (1, 1), activation='sigmoid',name='xoutput1')(xout6)

    xup6 = Dropout(DropP)(xup6)

    xconv6a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup6)
    
    xconv6a = bn()(xconv6a)

    merge0=concatenate([xup6,xconv6a])

    xconv6b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xconv6b = bn()(xconv6b)

    merge1=concatenate([xup6,xconv6a,xconv6b])

    xconv6c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xconv6c = bn()(xconv6c)

    merge2=concatenate([xup6,xconv6a,xconv6b,xconv6c])

    xconv6d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xconv6d = bn()(xconv6d)


    merge3=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d])



    xconv6e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xconv6e = bn()(xconv6e)

    merge4=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e])


    xconv6f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xconv6f = bn()(xconv6f)


    merge5=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f])

    xconv6g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xconv6g = bn()(xconv6g)

    merge6=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g])


    xconv6h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xconv6h = bn()(xconv6h)

    merge7=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h])


    xconv6i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xconv6i = bn()(xconv6i)

    merge8=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i])

    xconv6j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xconv6j = bn()(xconv6j)


    merge9=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j])


    xconv6k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xconv6k = bn()(xconv6k)


    merge10=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k])
    xconv6l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xconv6l = bn()(xconv6l)

    merge11=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l])
    xconv6m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xconv6m = bn()(xconv6m)


    merge12=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m])
    xconv6n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xconv6n = bn()(xconv6n)




    merge13=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n])
    xconv6o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xconv6o = bn()(xconv6o)



    merge14=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n,xconv6o])
    xconv6p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xconv6p = bn()(xconv6p)


    merge15=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n,xconv6o,xconv6p])
    xconv6q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xconv6q = bn()(xconv6q)


    merge16=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n,xconv6o,xconv6p,xconv6q])
    xconv6r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xconv6r = bn()(xconv6r)


    merge17=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n,xconv6o,xconv6p,xconv6q,xconv6r])
    xconv6s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xconv6s = bn()(xconv6s)


    merge18=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n,xconv6o,xconv6p,xconv6q,xconv6r,xconv6s])
    xconv6t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xconv6t = bn()(xconv6t)

    merge19=concatenate([xup6,xconv6a,xconv6b,xconv6c,xconv6d,xconv6e,xconv6f,xconv6g,xconv6h,xconv6i,xconv6j,xconv6k,xconv6l,xconv6m,xconv6n,xconv6o,xconv6p,xconv6q,xconv6r,xconv6s,xconv6t])
    xconv6u=Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xconv6u = bn()(xconv6u)





    xup7 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xconv6u), conv3u],name='xup7', axis=3)

    xup7 = Dropout(DropP)(xup7)
    #add second xoutput here
    xout7=Conv2DTranspose(12,(2, 2), strides=(4, 4), padding='same')(xup7)
    xout7 = bn()(xout7)
    xoutput2 = Conv2D(1, (1, 1), activation='sigmoid',name='xoutput2')(xout7)
    xconv7a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup7)
    
    xconv7a = bn()(xconv7a)

    merge0=concatenate([xup7,xconv7a])

    xconv7b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xconv7b = bn()(xconv7b)

    merge1=concatenate([xup7,xconv7a,xconv7b])

    xconv7c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xconv7c = bn()(xconv7c)

    merge2=concatenate([xup7,xconv7a,xconv7b,xconv7c])

    xconv7d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xconv7d = bn()(xconv7d)


    merge3=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d])



    xconv7e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xconv7e = bn()(xconv7e)

    merge4=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e])


    xconv7f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xconv7f = bn()(xconv7f)


    merge5=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f])

    xconv7g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xconv7g = bn()(xconv7g)

    merge6=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g])


    xconv7h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xconv7h = bn()(xconv7h)

    merge7=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h])


    xconv7i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xconv7i = bn()(xconv7i)

    merge8=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i])

    xconv7j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xconv7j = bn()(xconv7j)


    merge9=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j])


    xconv7k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xconv7k = bn()(xconv7k)


    merge10=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k])
    xconv7l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xconv7l = bn()(xconv7l)

    merge11=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l])
    xconv7m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xconv7m = bn()(xconv7m)


    merge12=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m])
    xconv7n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xconv7n = bn()(xconv7n)




    merge13=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n])
    xconv7o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xconv7o = bn()(xconv7o)



    merge14=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n,xconv7o])
    xconv7p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xconv7p = bn()(xconv7p)


    merge15=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n,xconv7o,xconv7p])
    xconv7q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xconv7q = bn()(xconv7q)


    merge16=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n,xconv7o,xconv7p,xconv7q])
    xconv7r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xconv7r = bn()(xconv7r)


    merge17=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n,xconv7o,xconv7p,xconv7q,xconv7r])
    xconv7s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xconv7s = bn()(xconv7s)


    merge18=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n,xconv7o,xconv7p,xconv7q,xconv7r,xconv7s])
    xconv7t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xconv7t = bn()(xconv7t)

    merge19=concatenate([xup7,xconv7a,xconv7b,xconv7c,xconv7d,xconv7e,xconv7f,xconv7g,xconv7h,xconv7i,xconv7j,xconv7k,xconv7l,xconv7m,xconv7n,xconv7o,xconv7p,xconv7q,xconv7r,xconv7s,xconv7t])
    xconv7u=Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xconv7u = bn()(xconv7u)




    xup8 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xconv7u), conv2j],name='xup8', axis=3)

    xup8 = Dropout(DropP)(xup8)
    #add third xoutxout here
    xout8=Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xup8)
    xout8 = bn()(xout8)
    xoutput3 = Conv2D(1, (1, 1), activation='sigmoid',name='xoutput3')(xout8)
    xconv8a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup8)
    
    xconv8a = bn()(xconv8a)

    merge0=concatenate([xup8,xconv8a])

    xconv8b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xconv8b = bn()(xconv8b)

    merge1=concatenate([xup8,xconv8a,xconv8b])

    xconv8c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xconv8c = bn()(xconv8c)

    merge2=concatenate([xup8,xconv8a,xconv8b,xconv8c])

    xconv8d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xconv8d = bn()(xconv8d)


    merge3=concatenate([xup8,xconv8a,xconv8b,xconv8c,xconv8d])



    xconv8e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xconv8e = bn()(xconv8e)

    merge4=concatenate([xup8,xconv8a,xconv8b,xconv8c,xconv8d,xconv8e])


    xconv8f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xconv8f = bn()(xconv8f)


    merge5=concatenate([xup8,xconv8a,xconv8b,xconv8c,xconv8d,xconv8e,xconv8f])

    xconv8g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xconv8g = bn()(xconv8g)

    merge6=concatenate([xup8,xconv8a,xconv8b,xconv8c,xconv8d,xconv8e,xconv8f,xconv8g])


    xconv8h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xconv8h = bn()(xconv8h)

    merge7=concatenate([xup8,xconv8a,xconv8b,xconv8c,xconv8d,xconv8e,xconv8f,xconv8g,xconv8h])


    xconv8i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xconv8i = bn()(xconv8i)

    merge8=concatenate([xup8,xconv8a,xconv8b,xconv8c,xconv8d,xconv8e,xconv8f,xconv8g,xconv8h,xconv8i])

    xconv8j = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xconv8j = bn()(xconv8j)


    
    xup9 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xconv8j), conv1d],name='xup9',axis=3)

    xup9 = Dropout(DropP)(xup9)
    xout9=Conv2DTranspose(12,(2, 2), strides=(1, 1), padding='same')(xup9)
    xout9 = bn()(xout9)
    xoutput4 = Conv2D(1, (1, 1), activation='sigmoid',name='xoutput4')(xout9)

    xconv9a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xup9)
    
    xconv9a = bn()(xconv9a)

    merge0=concatenate([xup9,xconv9a])

    xconv9b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xconv9b = bn()(xconv9b)

    merge1=concatenate([xup9,xconv9a,xconv9b])

    xconv9c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xconv9c = bn()(xconv9c)

    merge2=concatenate([xup9,xconv9a,xconv9b,xconv9c])

    xconv9d = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xconv9d = bn()(xconv9d)

   
    xconv10 = Conv2D(1, (1, 1), activation='sigmoid',name='xconv10')(xconv9d)

    xfinalmerge=concatenate([xout6,xout7,xout8,xout9,xconv9d])

    xfinal_op=Conv2D(1, (1, 1), activation='sigmoid',name='xfinal_op')(xfinalmerge)


    u_net_op0=keras.layers.add([final_op,xfinal_op])
    u_net_op1=keras.layers.add([conv10,xconv10])
    u_net_op2=keras.layers.add([output4,xoutput4])
    u_net_op3=keras.layers.add([output3,xoutput3])
    u_net_op4=keras.layers.add([output2,xoutput2])
    u_net_op5=keras.layers.add([output1,xoutput1])


    u_net_op_merge=concatenate([u_net_op0,u_net_op1,u_net_op2,u_net_op3,u_net_op4,u_net_op5])






    xxconv1a = Conv2D( 12, (kernel_size, kernel_size), activation='relu', padding='same', 
                   kernel_regularizer=regularizers.l2(l2_lambda) )(u_net_op_merge)
    
    
    xxconv1a = bn()(xxconv1a)
    
    xxconv1b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxconv1a)

    xxconv1b = bn()(xxconv1b)

    merge1=concatenate([xxconv1a,xxconv1b])
    
    xxconv1c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv1c = bn()(xxconv1c)

    merge2=concatenate([xxconv1a,xxconv1b,xxconv1c])

    xxconv1d = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv1d = bn()(xxconv1d)

    
    xxpool1 = MaxPooling2D(pool_size=(2, 2))(xxconv1d)

    xxpool1 = Dropout(DropP)(xxpool1)



    

    xxconv2a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxpool1)
    
    xxconv2a = bn()(xxconv2a)

    xxconv2b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxconv2a)

    xxconv2b = bn()(xxconv2b)

    merge1=concatenate([xxconv2a,xxconv2b])

    xxconv2c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv2c = bn()(xxconv2c)

    merge2=concatenate([xxconv2a,xxconv2b,xxconv2c])

    xxconv2d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv2d = bn()(xxconv2d)


    merge3=concatenate([xxconv2a,xxconv2b,xxconv2c,xxconv2d])



    xxconv2e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv2e = bn()(xxconv2e)

    merge4=concatenate([xxconv2a,xxconv2b,xxconv2c,xxconv2d,xxconv2e])


    xxconv2f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv2f = bn()(xxconv2f)


    merge5=concatenate([xxconv2a,xxconv2b,xxconv2c,xxconv2d,xxconv2e,xxconv2f])

    xxconv2g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv2g = bn()(xxconv2g)

    merge6=concatenate([xxconv2a,xxconv2b,xxconv2c,xxconv2d,xxconv2e,xxconv2f,xxconv2g])


    xxconv2h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv2h = bn()(xxconv2h)

    merge7=concatenate([xxconv2a,xxconv2b,xxconv2c,xxconv2d,xxconv2e,xxconv2f,xxconv2g,xxconv2h])


    xxconv2i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv2i = bn()(xxconv2g)

    merge8=concatenate([xxconv2a,xxconv2b,xxconv2c,xxconv2d,xxconv2e,xxconv2f,xxconv2g,xxconv2h,xxconv2i])

    xxconv2j = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv2j = bn()(xxconv2g)

    
    xxpool2 = MaxPooling2D(pool_size=(2, 2))(xxconv2j)

    xxpool2 = Dropout(DropP)(xxpool2)







    xxconv3a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxpool2)
    
    xxconv3a = bn()(xxconv3a)

    xxconv3b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxconv3a)

    xxconv3b = bn()(xxconv3b)

    merge1=concatenate([xxconv3a,xxconv3b])

    xxconv3c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv3c = bn()(xxconv3c)

    merge2=concatenate([xxconv3a,xxconv3b,xxconv3c])

    xxconv3d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv3d = bn()(xxconv3d)


    merge3=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d])



    xxconv3e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv3e = bn()(xxconv3e)

    merge4=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e])


    xxconv3f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv3f = bn()(xxconv3f)


    merge5=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f])

    xxconv3g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv3g = bn()(xxconv3g)

    merge6=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g])


    xxconv3h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv3h = bn()(xxconv3h)

    merge7=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h])


    xxconv3i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv3i = bn()(xxconv3i)

    merge8=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i])

    xxconv3j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv3j = bn()(xxconv3j)


    merge9=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j])


    xxconv3k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xxconv3k = bn()(xxconv3k)


    merge10=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k])
    xxconv3l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xxconv3l = bn()(xxconv3l)

    merge11=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l])
    xxconv3m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xxconv3m = bn()(xxconv3m)


    merge12=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m])
    xxconv3n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xxconv3n = bn()(xxconv3n)




    merge13=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n])
    xxconv3o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xxconv3o = bn()(xxconv3o)



    merge14=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n,xxconv3o])
    xxconv3p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xxconv3p = bn()(xxconv3p)


    merge15=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n,xxconv3o,xxconv3p])
    xxconv3q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xxconv3q = bn()(xxconv3q)


    merge16=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n,xxconv3o,xxconv3p,xxconv3q])
    xxconv3r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xxconv3r = bn()(xxconv3r)


    merge17=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n,xxconv3o,xxconv3p,xxconv3q,xxconv3r])
    xxconv3s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xxconv3s = bn()(xxconv3s)


    merge18=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n,xxconv3o,xxconv3p,xxconv3q,xxconv3r,xxconv3s])
    xxconv3t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xxconv3t = bn()(xxconv3t)

    merge19=concatenate([xxconv3a,xxconv3b,xxconv3c,xxconv3d,xxconv3e,xxconv3f,xxconv3g,xxconv3h,xxconv3i,xxconv3j,xxconv3k,xxconv3l,xxconv3m,xxconv3n,xxconv3o,xxconv3p,xxconv3q,xxconv3r,xxconv3s,xxconv3t])
    xxconv3u=Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xxconv3u = bn()(xxconv3u)


    xxpool3 = MaxPooling2D(pool_size=(2, 2))(xxconv3u)

    xxpool3 = Dropout(DropP)(xxpool3)

    
    xxconv4a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxpool3)
    
    xxconv4a = bn()(xxconv4a)

    xxconv4b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxconv4a)

    xxconv4b = bn()(xxconv4b)

    merge1=concatenate([xxconv4a,xxconv4b])

    xxconv4c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv4c = bn()(xxconv4c)

    merge2=concatenate([xxconv4a,xxconv4b,xxconv4c])

    xxconv4d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv4d = bn()(xxconv4d)


    merge3=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d])



    xxconv4e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv4e = bn()(xxconv4e)

    merge4=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e])


    xxconv4f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv4f = bn()(xxconv4f)


    merge5=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f])

    xxconv4g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv4g = bn()(xxconv4g)

    merge6=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g])


    xxconv4h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv4h = bn()(xxconv4h)

    merge7=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h])


    xxconv4i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv4i = bn()(xxconv4i)

    merge8=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i])

    xxconv4j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv4j = bn()(xxconv4j)


    merge9=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j])


    xxconv4k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xxconv4k = bn()(xxconv4k)


    merge10=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k])
    xxconv4l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xxconv4l = bn()(xxconv4l)

    merge11=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l])
    xxconv4m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xxconv4m = bn()(xxconv4m)


    merge12=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m])
    xxconv4n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xxconv4n = bn()(xxconv4n)




    merge13=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n])
    xxconv4o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xxconv4o = bn()(xxconv4o)



    merge14=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n,xxconv4o])
    xxconv4p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xxconv4p = bn()(xxconv4p)


    merge15=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n,xxconv4o,xxconv4p])
    xxconv4q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xxconv4q = bn()(xxconv4q)


    merge16=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n,xxconv4o,xxconv4p,xxconv4q])
    xxconv4r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xxconv4r = bn()(xxconv4r)


    merge17=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n,xxconv4o,xxconv4p,xxconv4q,xxconv4r])
    xxconv4s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xxconv4s = bn()(xxconv4s)


    merge18=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n,xxconv4o,xxconv4p,xxconv4q,xxconv4r,xxconv4s])
    xxconv4t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xxconv4t = bn()(xxconv4t)

    merge19=concatenate([xxconv4a,xxconv4b,xxconv4c,xxconv4d,xxconv4e,xxconv4f,xxconv4g,xxconv4h,xxconv4i,xxconv4j,xxconv4k,xxconv4l,xxconv4m,xxconv4n,xxconv4o,xxconv4p,xxconv4q,xxconv4r,xxconv4s,xxconv4t])
    xxconv4u=Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xxconv4u = bn()(xxconv4u)
    
    xxpool4 = MaxPooling2D(pool_size=(2, 2))(xxconv4u)

    xxpool4 = Dropout(DropP)(xxpool4)





    xxconv5a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxpool4)
    
    xxconv5a = bn()(xxconv5a)

    xxconv5b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxconv5a)

    xxconv5b = bn()(xxconv5b)

    merge1=concatenate([xxconv5a,xxconv5b])

    xxconv5c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv5c = bn()(xxconv5c)

    merge2=concatenate([xxconv5a,xxconv5b,xxconv5c])

    xxconv5d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv5d = bn()(xxconv5d)


    merge3=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d])



    xxconv5e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv5e = bn()(xxconv5e)

    merge4=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e])


    xxconv5f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv5f = bn()(xxconv5f)


    merge5=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f])

    xxconv5g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv5g = bn()(xxconv5g)

    merge6=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g])


    xxconv5h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv5h = bn()(xxconv5h)

    merge7=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h])


    xxconv5i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv5i = bn()(xxconv5i)

    merge8=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i])

    xxconv5j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv5j = bn()(xxconv5j)


    merge9=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j])


    xxconv5k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xxconv5k = bn()(xxconv5k)


    merge10=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k])
    xxconv5l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xxconv5l = bn()(xxconv5l)

    merge11=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l])
    xxconv5m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xxconv5m = bn()(xxconv5m)


    merge12=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m])
    xxconv5n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xxconv5n = bn()(xxconv5n)




    merge13=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n])
    xxconv5o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xxconv5o = bn()(xxconv5o)



    merge14=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n,xxconv5o])
    xxconv5p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xxconv5p = bn()(xxconv5p)


    merge15=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n,xxconv5o,xxconv5p])
    xxconv5q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xxconv5q = bn()(xxconv5q)


    merge16=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n,xxconv5o,xxconv5p,xxconv5q])
    xxconv5r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xxconv5r = bn()(xxconv5r)


    merge17=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n,xxconv5o,xxconv5p,xxconv5q,xxconv5r])
    xxconv5s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xxconv5s = bn()(xxconv5s)


    merge18=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n,xxconv5o,xxconv5p,xxconv5q,xxconv5r,xxconv5s])
    xxconv5t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xxconv5t = bn()(xxconv5t)

    merge19=concatenate([xxconv5a,xxconv5b,xxconv5c,xxconv5d,xxconv5e,xxconv5f,xxconv5g,xxconv5h,xxconv5i,xxconv5j,xxconv5k,xxconv5l,xxconv5m,xxconv5n,xxconv5o,xxconv5p,xxconv5q,xxconv5r,xxconv5s,xxconv5t])
    xxconv5u=Conv2D(512, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xxconv5u = bn()(xxconv5u)
    



    xxup6 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xxconv5u), xxconv4u],name='xxup6', axis=3)

    xxout6=Conv2DTranspose(12,(2, 2), strides=(8, 8), padding='same')(xxup6)
    xxout6 = bn()(xxout6)
    xxoutput1 = Conv2D(1, (1, 1), activation='sigmoid',name='xxoutput1')(xxout6)

    xxup6 = Dropout(DropP)(xxup6)

    xxconv6a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxup6)
    
    xxconv6a = bn()(xxconv6a)

    merge0=concatenate([xxup6,xxconv6a])

    xxconv6b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xxconv6b = bn()(xxconv6b)

    merge1=concatenate([xxup6,xxconv6a,xxconv6b])

    xxconv6c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv6c = bn()(xxconv6c)

    merge2=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c])

    xxconv6d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv6d = bn()(xxconv6d)


    merge3=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d])


    xxconv6e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv6e = bn()(xxconv6e)

    merge4=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e])


    xxconv6f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv6f = bn()(xxconv6f)


    merge5=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f])

    xxconv6g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv6g = bn()(xxconv6g)

    merge6=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g])


    xxconv6h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv6h = bn()(xxconv6h)

    merge7=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h])


    xxconv6i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv6i = bn()(xxconv6i)

    merge8=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i])

    xxconv6j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv6j = bn()(xxconv6j)


    merge9=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j])


    xxconv6k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xxconv6k = bn()(xxconv6k)


    merge10=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k])
    xxconv6l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xxconv6l = bn()(xxconv6l)

    merge11=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l])
    xxconv6m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xxconv6m = bn()(xxconv6m)


    merge12=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m])
    xxconv6n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xxconv6n = bn()(xxconv6n)




    merge13=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n])
    xxconv6o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xxconv6o = bn()(xxconv6o)



    merge14=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n,xxconv6o])
    xxconv6p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xxconv6p = bn()(xxconv6p)


    merge15=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n,xxconv6o,xxconv6p])
    xxconv6q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xxconv6q = bn()(xxconv6q)


    merge16=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n,xxconv6o,xxconv6p,xxconv6q])
    xxconv6r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xxconv6r = bn()(xxconv6r)


    merge17=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n,xxconv6o,xxconv6p,xxconv6q,xxconv6r])
    xxconv6s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xxconv6s = bn()(xxconv6s)


    merge18=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n,xxconv6o,xxconv6p,xxconv6q,xxconv6r,xxconv6s])
    xxconv6t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xxconv6t = bn()(xxconv6t)

    merge19=concatenate([xxup6,xxconv6a,xxconv6b,xxconv6c,xxconv6d,xxconv6e,xxconv6f,xxconv6g,xxconv6h,xxconv6i,xxconv6j,xxconv6k,xxconv6l,xxconv6m,xxconv6n,xxconv6o,xxconv6p,xxconv6q,xxconv6r,xxconv6s,xxconv6t])
    xxconv6u=Conv2D(256, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xxconv6u = bn()(xxconv6u)





    xxup7 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xxconv6u), xxconv3u],name='xxup7', axis=3)

    xxup7 = Dropout(DropP)(xxup7)
    #add second xxoutput here
    xxout7=Conv2DTranspose(12,(2, 2), strides=(4, 4), padding='same')(xxup7)
    xxout7 = bn()(xxout7)
    xxoutput2 = Conv2D(1, (1, 1), activation='sigmoid',name='xxoutput2')(xxout7)
    xxconv7a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxup7)
    
    xxconv7a = bn()(xxconv7a)

    merge0=concatenate([xxup7,xxconv7a])

    xxconv7b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xxconv7b = bn()(xxconv7b)

    merge1=concatenate([xxup7,xxconv7a,xxconv7b])

    xxconv7c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv7c = bn()(xxconv7c)

    merge2=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c])

    xxconv7d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv7d = bn()(xxconv7d)


    merge3=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d])



    xxconv7e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv7e = bn()(xxconv7e)

    merge4=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e])


    xxconv7f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv7f = bn()(xxconv7f)


    merge5=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f])

    xxconv7g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv7g = bn()(xxconv7g)

    merge6=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g])


    xxconv7h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv7h = bn()(xxconv7h)

    merge7=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h])


    xxconv7i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv7i = bn()(xxconv7i)

    merge8=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i])

    xxconv7j = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv7j = bn()(xxconv7j)


    merge9=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j])


    xxconv7k = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)

    xxconv7k = bn()(xxconv7k)


    merge10=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k])
    xxconv7l=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
    xxconv7l = bn()(xxconv7l)

    merge11=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l])
    xxconv7m=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
    xxconv7m = bn()(xxconv7m)


    merge12=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m])
    xxconv7n=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
    xxconv7n = bn()(xxconv7n)




    merge13=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n])
    xxconv7o=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
    xxconv7o = bn()(xxconv7o)



    merge14=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n,xxconv7o])
    xxconv7p=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
    xxconv7p = bn()(xxconv7p)


    merge15=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n,xxconv7o,xxconv7p])
    xxconv7q=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
    xxconv7q = bn()(xxconv7q)


    merge16=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n,xxconv7o,xxconv7p,xxconv7q])
    xxconv7r=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
    xxconv7r = bn()(xxconv7r)


    merge17=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n,xxconv7o,xxconv7p,xxconv7q,xxconv7r])
    xxconv7s=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
    xxconv7s = bn()(xxconv7s)


    merge18=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n,xxconv7o,xxconv7p,xxconv7q,xxconv7r,xxconv7s])
    xxconv7t=Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
    xxconv7t = bn()(xxconv7t)

    merge19=concatenate([xxup7,xxconv7a,xxconv7b,xxconv7c,xxconv7d,xxconv7e,xxconv7f,xxconv7g,xxconv7h,xxconv7i,xxconv7j,xxconv7k,xxconv7l,xxconv7m,xxconv7n,xxconv7o,xxconv7p,xxconv7q,xxconv7r,xxconv7s,xxconv7t])
    xxconv7u=Conv2D(128, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
    xxconv7u = bn()(xxconv7u)




    xxup8 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xxconv7u), xxconv2j],name='xxup8', axis=3)

    xxup8 = Dropout(DropP)(xxup8)
    #add third xxoutxxout here
    xxout8=Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xxup8)
    xxout8 = bn()(xxout8)
    xxoutput3 = Conv2D(1, (1, 1), activation='sigmoid',name='xxoutput3')(xxout8)
    xxconv8a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxup8)
    
    xxconv8a = bn()(xxconv8a)

    merge0=concatenate([xxup8,xxconv8a])

    xxconv8b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xxconv8b = bn()(xxconv8b)

    merge1=concatenate([xxup8,xxconv8a,xxconv8b])

    xxconv8c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv8c = bn()(xxconv8c)

    merge2=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c])

    xxconv8d = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv8d = bn()(xxconv8d)


    merge3=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c,xxconv8d])



    xxconv8e = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)

    xxconv8e = bn()(xxconv8e)

    merge4=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c,xxconv8d,xxconv8e])


    xxconv8f = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)

    xxconv8f = bn()(xxconv8f)


    merge5=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c,xxconv8d,xxconv8e,xxconv8f])

    xxconv8g = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)

    xxconv8g = bn()(xxconv8g)

    merge6=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c,xxconv8d,xxconv8e,xxconv8f,xxconv8g])


    xxconv8h = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)

    xxconv8h = bn()(xxconv8h)

    merge7=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c,xxconv8d,xxconv8e,xxconv8f,xxconv8g,xxconv8h])


    xxconv8i = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)

    xxconv8i = bn()(xxconv8i)

    merge8=concatenate([xxup8,xxconv8a,xxconv8b,xxconv8c,xxconv8d,xxconv8e,xxconv8f,xxconv8g,xxconv8h,xxconv8i])

    xxconv8j = Conv2D(64, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)

    xxconv8j = bn()(xxconv8j)


    
    xxup9 = concatenate([Conv2DTranspose(12,(2, 2), strides=(2, 2), padding='same')(xxconv8j), xxconv1d],name='xxup9',axis=3)

    xxup9 = Dropout(DropP)(xxup9)
    xxout9=Conv2DTranspose(12,(2, 2), strides=(1, 1), padding='same')(xxup9)
    xxout9 = bn()(xxout9)
    xxoutput4 = Conv2D(1, (1, 1), activation='sigmoid',name='xxoutput4')(xxout9)

    xxconv9a = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(xxup9)
    
    xxconv9a = bn()(xxconv9a)

    merge0=concatenate([xxup9,xxconv9a])

    xxconv9b = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge0)

    xxconv9b = bn()(xxconv9b)

    merge1=concatenate([xxup9,xxconv9a,xxconv9b])

    xxconv9c = Conv2D(12, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)

    xxconv9c = bn()(xxconv9c)

    merge2=concatenate([xxup9,xxconv9a,xxconv9b,xxconv9c])

    xxconv9d = Conv2D(32, (kernel_size, kernel_size), activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)

    xxconv9d = bn()(xxconv9d)

   
    xxconv10 = Conv2D(1, (1, 1), activation='sigmoid',name='xxconv10')(xxconv9d)

    xxfinalmerge=concatenate([xxout6,xxout7,xxout8,xxout9,xxconv9d])

    xxfinal_op=Conv2D(1, (1, 1), activation='sigmoid',name='xxfinal_op')(xxfinalmerge)

    #model = Model(inputs=[inputs,input_prob,input_prob_inverse], outputs=[conv10,xconv10,third_out])
    model = Model(inputs=inputs, outputs=[output1,output2,output3,output4,conv10,final_op,xoutput1,xoutput2,xoutput3,xoutput4,xconv10,xfinal_op,xxoutput1,xxoutput2,xxoutput3,xxoutput4,xxconv10,xxfinal_op])

    model.compile(optimizer=Adam(lr=1e-5), loss={'output1':dice_coef_loss,'output2':dice_coef_loss,'output3':dice_coef_loss,'output4':dice_coef_loss,'conv10':dice_coef_loss,'final_op':dice_coef_loss,
                                                'xoutput1':neg_dice_coef_loss,'xoutput2':neg_dice_coef_loss,'xoutput3':neg_dice_coef_loss,'xoutput4':neg_dice_coef_loss,'xconv10':neg_dice_coef_loss,'xfinal_op':neg_dice_coef_loss,
                                                'xxoutput1':'mse','xxoutput2':'mse','xxoutput3':'mse','xxoutput4':'mse','xxconv10':'mse','xxfinal_op':'mse'})
      #loss=[neg_dice_coef_loss,'mse',dice_coef_loss],
       #metrics=[neg_dice_coef,'mae',dice_coef])
    return model
 

# In[8]:


model=UNet(input_shape=(512,512,1))
print(model.summary())


# In[62]:
X_train=np.load("X_train_new.npy")
X_train=X_train.reshape(X_train.shape+(1,))
y_train=np.load("y_train_new.npy").reshape(X_train.shape)

model.fit([X_train], [y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,y_train,X_train,X_train,X_train,X_train,X_train,X_train],
                  batch_size=1,
                  nb_epoch=2,
                        #validation_data=([X2_validate],[y_validate]),
                  shuffle=True)
                        #callbacks=[xyz],
                        #class_weight=class_weightt)


# In[29]:



import h5py
#model.save_weights("basic_unet_weights.h5")
model.save('dense_comp_net_dsp.h5')


'''
for j in range(5,9):

  X_train=(np.load("level"+str(xx)+"/X.npy"))
  X_train=X_train.reshape((X_train.shape)+(1,))

  X1_train=(np.load("level"+str(xx)+"/prob_mask.npy")).reshape((X_train.shape))
  y_train=(np.load("level"+str(xx)+"/y.npy")).reshape((X_train.shape))
  y1_train=(np.load("level"+str(xx)+"/skull_mask.npy")).reshape((X_train.shape))
  X2_train=np.load(("level"+str(xx)+"/prob_mask_inverse.npy")).reshape((X_train.shape))
  y_third=np.load(("level"+str(xx)+"/total_mask.npy")).reshape((X_train.shape))

# In[30]:



  predicted=model.predict([X_train,X1_train,X2_train],batch_size=8)[1]
  print('done')


# In[35]:


  print(predicted.shape)


# In[36]:


#import cv2
#for i in range(0,len(X_test)):
#    cv2.imwrite("small_section_output/"+str(i)+".png",y_test[i])
#    cv2.imwrite("small_section_predict/"+str(i)+".png",predicted[i])
#print ("done")
    


# In[49]:


  smooth=1
  def dice_calc(y_true, y_pred):
      y_true_f = y_true.flatten()
      y_pred_f = y_pred.flatten()
      intersection = sum(y_true_f * y_pred_f)
      return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)


  # In[50]:


  y_test=np.array(y_test)
  predicted=np.array(predicted)


  # In[51]:


  dice=0
  for i in range(0,len(X_train)):
      dice=dice+dice_calc(y1_train[i],predicted[i])
  print(dice)
      


  # In[48]:


  print (dice/len(X_test))

'''