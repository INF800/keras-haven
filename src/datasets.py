from haven import haven_utils as hu
from haven import haven_wizard as hw

import numpy as np
from tensorflow import keras
import tensorflow as tf


def get_loader(name, split, datadir, exp_dict, download=True):
    """
    :param name: {string}. name of the dataset
    :param split: {str}. 'train', 'val', 'test'
    :param datadir: {str|None}. default `None` path to the dataset directory. Data will be downloaded to this path if `download` is set to `True` and data does not exist.
    :param exp_dict: {dict}. dictionary with the experiment parameters
    :param download: {bool}. default `True`. download the dataset if it doesn't exist
    """

    if name == "mnist":
        num_classes = 10

        # get dataset. Automatically download if not exist into some other dir instead of `datadir`.
        (x_data, y_data), (x_test, y_test) = keras.datasets.mnist.load_data()

        # split data so that validation and test size are same.
        x_train, x_valid = x_data[:50000], x_data[50000:]
        y_train, y_valid = y_data[:50000], y_data[50000:]

        x_train =  np.expand_dims(x_train, -1)
        x_valid = np.expand_dims(x_valid, -1)
        x_test =  np.expand_dims(x_test, -1)

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_valid = keras.utils.to_categorical(y_valid, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        # we can create either custom data generator or use the default one.
        # one advantage with this data generator is inbuilt data augmentation.
        # can use `flow` or `flow_from_directory` or `flow_from_dataframe`.
        gen = tf.keras.preprocessing.image.ImageDataGenerator(**exp_dict['augmentations'])#(rescale=1/255.)

        # return data generators / loaders
        if split=='train':
            return gen.flow(x_train, y_train, batch_size=exp_dict['batch_size'], seed=exp_dict['seed']) # shuffle is set to True by default.
        if split=='val':
            return gen.flow(x_valid, y_valid, batch_size=exp_dict['batch_size'], seed=exp_dict['seed'])
        if split=='test':
            return gen.flow(x_test, y_test, batch_size=exp_dict['batch_size'], seed=exp_dict['seed'])

