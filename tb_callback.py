import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import io
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train', 'test'],
                                         shuffle_files=True,
                                         as_supervised=True, with_info=True)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


def augment(image, label):
    if tf.random.uniform((), minval=0, maxval=1) < 0.1:
        # Convert RGB to Gray and keep the original number of channels
        image = tf.tile(tf.image.rgb_to_grayscale(image), [1, 1, 3])

    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.2)

    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

# setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# setup for test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]


def get_model():
    model = keras.Sequential([
        keras.Input((32, 32, 3)),
        layers.Conv2D(4, 3, padding='same', activation='relu'),
        layers.Conv2D(8, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    return model


model = get_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='tb_callback_dir', histogram_freq=1
)
model.fit(ds_train, epochs=5, validation_data=ds_test,
          callbacks=[tensorboard_callback], verbose=2)

# num_epochs = 1
# loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# optimizer = keras.optimizers.Adam()
# acc_metric = keras.metrics.SparseCategoricalAccuracy()
# train_writer = tf.summary.create_file_writer('logs/train/')
# test_writer = tf.summary.create_file_writer('logs/test')
# train_step = test_step = 0
