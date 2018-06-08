'''
Trains a GDenseNet-40-12 model on the CIFAR-10 Dataset.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras_gcnn.applications.densenetnew import GDenseNet

batch_size = 16
nb_classes = 10
epochs = 100

img_rows, img_cols = 32, 32
img_channels = 3

# Parameters for the DenseNet model builder
img_dim = (img_channels, img_rows, img_cols) if K.image_data_format() == 'channels_first' else (
    img_rows, img_cols, img_channels)
depth = 40
nb_dense_block = 3
growth_rate = 3  # number of z2 maps equals growth_rate * group_size, so keep this small.
nb_filter = 16
dropout_rate = 0.0  # 0.0 for data augmentation
conv_group = 'D4'  # C4 includes 90 degree rotations, D4 additionally includes reflections in x and y axis.
use_gcnn = True

# Create the model (without loading weights)
model = GDenseNet(mc_dropout=False, padding='same', depth=depth, nb_dense_block=nb_dense_block,
                  growth_rate=growth_rate, nb_filter=nb_filter,
                  dropout_rate=dropout_rate,
                  input_shape=img_dim,
                  weights=None, use_gcnn=use_gcnn, conv_group=conv_group)
print('Model created')

model.summary()

optimizer = Adam(lr=1e-3)  # Using Adam instead of SGD to speed up training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
print('Finished compiling')

(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype('float32')
testX = testX.astype('float32')

trainX /= 255.
testX /= 255.

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

# Test equivariance by comparing outputs for rotated versions of same datapoint:
res = model.predict(np.stack([trainX[123], np.rot90(trainX[123])]))
is_equivariant = np.allclose(res[0], res[1])
print('Equivariance check:', is_equivariant)
assert is_equivariant

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5. / 32,
                               height_shift_range=5. / 32)

generator.fit(trainX, seed=0)

weights_file = 'DenseNet-40-12-CIFAR-10.h5'

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                               cooldown=0, patience=10, min_lr=0.5e-6)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=20)
model_checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', save_best_only=True,
                                   save_weights_only=True, mode='auto')

callbacks = [lr_reducer, early_stopper, model_checkpoint]

model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=(testX, Y_test),
                    verbose=1)

scores = model.evaluate(testX, Y_test, batch_size=batch_size)
print('Test loss : ', scores[0])
print('Test accuracy : ', scores[1])
