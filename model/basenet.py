import tensorflow as tf
# from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.layers import Input, Conv2D, ReLU, Concatenate, Lambda, Layer, Add


def basenet(n_feat=64, out_c=3, scale_factor=3):
    x_in = Input(shape=(None, None, 3))

    w = np.transpose(np.array([[
    [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]]*scale_factor**2
    ]]), (0, 1, 3, 2))

    res_conv = Conv2D(out_c*(scale_factor**2), kernel_size=1, padding='same', use_bias=False)
    res_conv.trainable = False
    res_conv.build(x_in.shape)
    res_conv.set_weights([w])
    x_res = res_conv(x_in)
    
    x = Conv2D(n_feat, kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x_in)
    x = Conv2D(n_feat, kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(n_feat, kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(n_feat, kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(n_feat, kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(out_c*(scale_factor**2), kernel_size=3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(out_c*(scale_factor**2), kernel_size=3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Add()([x_res, x])
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor))
    x = depth_to_space(x)

    clip_func = Lambda(lambda x: tf.clip_by_value(x, 0., 255.))
    x = clip_func(x)

    return Model(x_in, x)


