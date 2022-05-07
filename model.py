from keras.layers import Activation, Convolution2D, Dropout, Conv2D, Dense, Concatenate, Add, PReLU, LeakyReLU, Multiply
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras import regularizers
from keras.activations import linear
from keras.layers import multiply
#from keras.layers import lambda
import tensorflow as tf
from san_model import gc, DR, CR, excitation, spatial_squeeze_excite_block, squeeze_excite_block
from keras_layer_normalization import LayerNormalization

import numpy as np
import cv2

num_features = 64
num_labels = 7


def SAN_CNN(input_shape, num_classes):

    
    img_input = Input(input_shape)

    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(img_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    


    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    

    input_x_2 = x
    split_2 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_x_2) 

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(split_2[1])
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)


    x = Concatenate()([x, split_2[0]])
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    
    input_x_3 = x
    split_3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_x_3) 

    x = Conv2D(320, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(split_3[1])
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(320, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)


    x = Concatenate()([x, split_3[0]]) 
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    input_x_4 = x
    split_4 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1))(input_x_4)     

    x = Conv2D(480, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(split_4[1])
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(480, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)


    x = Concatenate()([x, split_4[0]]) 

#---------------ATT-------------------------------------------------------------------------
    
    gc_1 = excitation(x, ratio=16)
    gc_2 = gc(x, ratio=48)
    a1 = Add()([gc_1, gc_2])  
    x = a1
    
    x = layers.Conv2D(64, (1, 1), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.LeakyReLU()(x)
    x = BatchNormalization()(x)    
    x = layers.Conv2D(720, (1, 1), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)   
    x = layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Add()([a1, x])
    
    
    cam = CR(x, ratio=16)    
    cam = Add()([a1, cam])
    x = cam


    x = layers.Conv2D(64, (1, 1), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.LeakyReLU()(x)
    x = BatchNormalization()(x)    
    x = layers.Conv2D(720, (1, 1), padding='same',kernel_regularizer=regularizers.l2(0.01))(x)   
    x = layers.LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)   
    x = Add()([cam, x])
    
    pam = DR(x, ratio=16)
    pam = Add()([cam, pam])
    x = pam

    
#--------------SPP-------------------------------------------------------------------------- 
    xp_1 = MaxPooling2D(pool_size=(2, 2), padding='same', strides=1)(x)
    xp_1 = Flatten()(xp_1)

   
    xp_2 = ZeroPadding2D(padding=1)(x) 
    xp_2 = MaxPooling2D(pool_size=(3, 3), strides=2)(xp_2) 
    xp_2 = Flatten()(xp_2)

    xp_3 = (x)     
    xp_3 = MaxPooling2D(pool_size=(3, 3))(xp_3)
    xp_3 = Flatten()(xp_3)
   
    x = Concatenate()([xp_3, xp_2, xp_1]) 

    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(img_input, x)
    return model
    


if __name__ == "__main__":
    input_shape = (48, 48, 1)
    num_classes = 7
    
