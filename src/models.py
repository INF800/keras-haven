import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def get_model(name, exp_dict):
    if name == "convnet_shallow":
        model = ConvnetShallow()
        
        # todo: set other optimizers
        if exp_dict["optimizer"]=="adam":
            optimizer =  tf.keras.optimizers.Adam(**exp_dict["optimizer_params"])
        
        model.compile(loss=exp_dict["loss"], optimizer=optimizer, metrics=exp_dict["metrics"])
    
    return model


class ConvnetShallow(tf.keras.Model):
    def __init__(self):
        super().__init__()

        input_shape = (28, 28, 1)
        num_classes = 10

        # it is recommended to build model as in: https://www.tensorflow.org/api_docs/python/tf/keras/Model
        # self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        # self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        
        # due to shortage of time, building model for single pass.
        self.full_model =  keras.Sequential([keras.Input(shape=input_shape),
                                             layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                                             layers.MaxPooling2D(pool_size=(2, 2)),
                                             layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                                             layers.MaxPooling2D(pool_size=(2, 2)),
                                             layers.Flatten(),
                                             layers.Dropout(0.5),
                                             layers.Dense(num_classes, activation="softmax"),])

    def call(self, inputs): 
        # x = self.dense1(inputs)
        # x = self.dense2(x)
        # return x
        return self.full_model(inputs)


