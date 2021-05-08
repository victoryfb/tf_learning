import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'],
                                         shuffle_files=True,
                                         as_supervised=True, with_info=True)


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 128

# setup for train dataset
ds_train = ds_train.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(BATCH_SIZE)
ds_train = ds_train.prefetch(AUTOTUNE)

# setup for test dataset
ds_test = ds_test.map(normalize_img, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.prefetch(AUTOTUNE)

model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, padding='same'),
    layers.ReLU(),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
], name='model')

num_epochs = 5
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Training loop
for epoch in range(num_epochs):
    print(f'\nStart of Training Epoch {epoch}')
    for batch_idx, (x_batch, y_batch) in enumerate(ds_train):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)

    training_acc = acc_metric.result()
    print(f'Accuracy over epoch {training_acc}')
    acc_metric.reset_states()

# Test loop
for batch_idx, (x_batch, y_batch) in enumerate(ds_test):
    y_pred = model(x_batch, training=False)
    acc_metric.update_state(y_batch, y_pred)

test_acc = acc_metric.result()
print(f'\nAccuracy over test data {test_acc}')
acc_metric.reset_states()
