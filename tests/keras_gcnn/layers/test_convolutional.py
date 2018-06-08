import numpy as np
import pytest
import tensorflow as tf

import groupy.garray.C4_array as C4a
import groupy.garray.D4_array as D4a
from groupy.gconv.keras_gconv.layers import GConv2D
from groupy.gfunc.p4func_array import P4FuncArray
from groupy.gfunc.p4mfunc_array import P4MFuncArray
from groupy.gfunc.z2func_array import Z2FuncArray


def test_c4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    x, y = make_graph('Z2', 'C4')
    equivariance_check(im, x, y, Z2FuncArray, P4FuncArray, C4a)


def test_c4_c4_conv_equivariance():
    im = np.random.randn(2, 5, 5, 4)
    x, y = make_graph('C4', 'C4')
    equivariance_check(im, x, y, P4FuncArray, P4FuncArray, C4a)


def test_d4_z2_conv_equivariance():
    im = np.random.randn(2, 5, 5, 1)
    x, y = make_graph('Z2', 'D4')
    equivariance_check(im, x, y, Z2FuncArray, P4MFuncArray, D4a)


def test_d4_d4_conv_equivariance():
    im = np.random.randn(2, 5, 5, 8)
    x, y = make_graph('D4', 'D4')
    equivariance_check(im, x, y, P4MFuncArray, P4MFuncArray, D4a)


def make_graph(h_input, h_output):
    l = GConv2D(1, 3, h_input, h_output)
    input_dim = 1
    if h_input == 'C4':
        input_dim *= 4
    elif h_input == 'D4':
        input_dim *= 8
    l.build([None, None, input_dim])
    nti = l.gconv_shape_info[-2]
    x = tf.placeholder(tf.float32, [None, 5, 5, 1 * nti])
    y = l(x)
    return x, y


def equivariance_check(im, input, output, input_array, output_array, point_group):
    # Transform the image
    f = input_array(im.transpose((0, 3, 1, 2)))
    g = point_group.rand()
    gf = g * f
    im1 = gf.v.transpose((0, 2, 3, 1))

    # Compute
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    yx = sess.run(output, feed_dict={input: im})
    yrx = sess.run(output, feed_dict={input: im1})
    sess.close()

    # Transform the computed feature maps
    fmap1_garray = output_array(yrx.transpose((0, 3, 1, 2)))
    r_fmap1_data = (g.inv() * fmap1_garray).v.transpose((0, 2, 3, 1))

    print(np.abs(yx - r_fmap1_data).sum())
    assert np.allclose(yx, r_fmap1_data, rtol=1e-5, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
