import numpy as np
import pytest
from numpy.testing import assert_allclose

from groupy.gconv.keras_gconv.layers import GBatchNorm
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.utils.test_utils import layer_test, keras_test

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


@keras_test
def test_basic_batchnorm():
    layer_test(GBatchNorm,
               kwargs={'h': 'D4',
                       'momentum': 0.9,
                       'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2 * 8))

    layer_test(GBatchNorm,
               kwargs={'h': 'C4',
                       'momentum': 0.9,
                       'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2 * 4))


@keras_test
def test_batchnorm_convnet():
    model = Sequential()
    norm = GBatchNorm(h='D4', axis=-1, input_shape=(4, 4, 3 * 8), momentum=0.8)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 4, 4, 3 * 8))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(K.eval(norm.repeated_beta), (1, 1, 1, 3 * 8))
    out /= np.reshape(K.eval(norm.repeated_gamma), (1, 1, 1, 3 * 8))

    assert_allclose(np.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 1, 2)), 1.0, atol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
