from keras import layers
from keras_layer_normalization import LayerNormalization
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
import tensorflow as tf
import keras.backend as K
from keras import backend as K


def squeeze_excite_block(input, ratio=16):

    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def spatial_squeeze_excite_block(input):

    se = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False,
                kernel_initializer='he_normal')(input)

    x = multiply([input, se])
    return x


def excitation(input, ratio=16):

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x




def matmul(input):
    """input must be a  list"""
    return tf.matmul(input[0],input[1])

def gc(x, ratio=16):

    bs, h, w, c = x.get_shape().as_list()

    input_x = layers.Conv2D(filters=c, kernel_size=(1, 1))(x) 
    input_x = layers.Reshape((h*w, c))(input_x)  
    input_x = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input_x) 
    input_x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(input_x)  

    context_mask = layers.Conv2D(filters=1, kernel_size=(1, 1))(x) 
    context_mask = layers.Reshape((h*w, 1))(context_mask)
    context_mask = layers.Softmax(axis=1)(context_mask)  
    context_mask = layers.Lambda(lambda x: tf.transpose(x, [0, 2, 1]))(context_mask) 
    context_mask = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(context_mask) 

    context = layers.Lambda(matmul)([input_x,context_mask])  
    context = layers.Reshape((1, 1, c))(context)

    context_transform = layers.Conv2D(c//ratio, (1, 1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = layers.ReLU()(context_transform)
    context_transform = layers.Conv2D(c, (1, 1), padding='same' )(context_transform)
    context_transform = LayerNormalization()(context_transform)
    context_transform = layers.ReLU()(context_transform)

    x = layers.Add()([x,context_transform])
   

    return x


def DR(x, ratio=32):

    bs, h, w, ch = x.get_shape().as_list()
    input_x = x
    b = (input_x)
    c = (input_x)
    d = (input_x)

    vec_b = layers.Reshape((h*w, ch))(b) 

    vec_cT = layers.Reshape((h*w, ch))(c)  
    vec_cT = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(vec_cT)  
    
    bcT = layers.Lambda(matmul)([vec_b,vec_cT])  

    softmax_bcT = layers.Softmax(axis=1)(bcT)  
    vec_d = layers.Reshape((h*w, ch))(d) 

    bcTd = layers.Lambda(matmul)([softmax_bcT,vec_d]) 

    bcTd = layers.Reshape((h, w, ch))(bcTd)  

    context = layers.Add()([bcTd, x]) 

    context_transform = layers.Conv2D(ch//ratio, (1, 1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = layers.ReLU()(context_transform)
    context_transform = layers.Dropout(0.5)(context_transform)
    x = layers.Conv2D(ch, (1, 1))(context_transform)

    return x

def CR(x, ratio=32):

    bs, h, w, ch = x.get_shape().as_list()
    input_x = x

    vec_a = layers.Reshape((h*w, ch))(input_x)  

    vec_aT = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(vec_a)  
    
    aTa = layers.Lambda(matmul)([vec_aT,vec_a])  

    softmax_aTa = layers.Softmax(axis=1)(aTa)  

    aaTa = layers.Lambda(matmul)([vec_a,softmax_aTa])  

    aaTa = layers.Reshape((h, w, ch))(aaTa) 

    context = layers.Add()([aaTa,x]) 

    context_transform = layers.Conv2D(ch//ratio, (1, 1))(context)
    context_transform = LayerNormalization()(context_transform)
    context_transform = layers.ReLU()(context_transform)
    context_transform = layers.Dropout(0.5)(context_transform)
    x = layers.Conv2D(ch, (1, 1))(context_transform)

    return x
