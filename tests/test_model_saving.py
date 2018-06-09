import os
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_allclose

from keras import backend as K
from keras import losses
from keras import metrics
from keras import optimizers
from keras.models import save_model, load_model
from keras.utils import np_utils
from keras.utils.test_utils import keras_test
from keras_gcnn.applications.densenetnew import GDenseNet

skipif_no_tf_gpu = pytest.mark.skipif(
    (K.backend() != 'tensorflow') or (not K.tensorflow_backend._get_available_gpus()),
    reason='Requires TensorFlow backend and a GPU')


@keras_test
def test_functional_model_saving():
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
    model = GDenseNet(mc_dropout=False, padding='same', nb_dense_block=nb_dense_block, growth_rate=growth_rate,
                      nb_filter=nb_filter, dropout_rate=dropout_rate, weights=None, input_shape=img_dim, depth=depth,
                      use_gcnn=use_gcnn, conv_group=conv_group)
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 32, 32, 3))
    y = np.random.randint(0, 10, 1)
    y = np_utils.to_categorical(y, 10)
    model.train_on_batch(x, y)

    out = model.predict(x)
    _, fname = tempfile.mkstemp('.h5')
    save_model(model, fname)

    model = load_model(fname)
    os.remove(fname)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


if __name__ == '__main__':
    pytest.main([__file__])
