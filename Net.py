import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Activation, Flatten, Dense
from tensorflow.keras import Model, models


def AlexNet8():
    # 第一层：卷积--> 激活 --> 归一化 --> 最大池化
    x = Conv2D(filters=96, kernel_size=11, strides=4, name='Conv2D_1')(inputs)
    x = Activation('relu', name='Activation_1')(x)
    x = BatchNormalization(name='BN_1')(x)  # 使用 BatchNormalization 代替LRN
    x = MaxPool2D(pool_size=(3, 3), strides=2, name='MaxPool2D_1')(x)

    # 第二层：卷积--> 激活 --> 归一化 --> 最大池化
    x = Conv2D(filters=256, kernel_size=5, strides=1, padding='same', name='Conv2D_2')(x)
    x = Activation('relu', name='Activation_2')(x)
    x = BatchNormalization(name='BN_2')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, name='MaxPool2D_2')(x)

    # 第三层：卷积--> 激活
    x = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', name='Conv2D_3')(x)
    x = Activation('relu', name='Activation_3')(x)

    # 第四层：卷积--> 激活
    x = Conv2D(filters=384, kernel_size=3, strides=1, padding='same', name='Conv2D_4')(x)
    x = Activation('relu', name='Activation_4')(x)

    # 第五层：卷积--> 激活  --> 最大池化
    x = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='Conv2D_5')(x)
    x = Activation('relu', name='Activation_5')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, name='MaxPool2D_5')(x)

    # 第六层  拉直 --> Drop out --> 激活
    x = Flatten()(x)
    x = Dense(units=4096, activation='relu', name='Dense_6')(x)
    x = Dropout(0.5, name='Dropout_6')(x)

    # 第七层 全连接 --> Drop out
    x = Dense(units=4096, activation='relu', name='Dense_7')(x)
    x = Dropout(0.5, name='Dropout_7')(x)

    # 第八层 全连接
    outputs = Dense(units=1000, activation='softmax', name='Output_8')(x)

    return outputs


# 第零层，输入层
inputs = tf.keras.Input(shape=(227, 227, 3))
model = Model(inputs=inputs, outputs=AlexNet8(), name='AlexNet8')
model.summary()