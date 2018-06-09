# -*- coding: utf-8 -*-
'''Group-Equivariant DenseNet for Keras.

# Reference
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
- []

This implementation is based on the following reference code:
 - https://github.com/gpleiss/efficient_densenet_pytorch
 - https://github.com/liuzhuang13/DenseNet

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input as _preprocess_input
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2
from keras_contrib.layers.convolutional import SubPixelUpscaling

from keras_gcnn.layers import GConv2D, GBatchNorm
from keras_gcnn.layers.pooling import GroupPool


def crop_to_fit(main, to_crop):
    from keras.layers import Cropping2D
    import keras.backend as K
    cropped_skip = to_crop
    skip_size = K.int_shape(cropped_skip)[1]
    out_size = K.int_shape(main)[1]
    if skip_size > out_size:
        size_diff = (skip_size - out_size) // 2
        size_diff_odd = ((skip_size - out_size) // 2) + ((skip_size - out_size) % 2)
        cropped_skip = Cropping2D(((size_diff, size_diff_odd),) * 2)(cropped_skip)
    return cropped_skip


def __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=0.99, epsilon=1e-3, center=True, scale=True,
                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                gamma_constraint=None, axis=-1, **kwargs):
    """Utility function to get batchnorm operation.

    # Arguments
        filters: filters in `Conv2D`
        kernel_size: height and width of the convolution kernel (tuple)
        strides: stride in 'Conv2D'
        padding: padding mode in `Conv2D`
        use_bias: bias mode in `Conv2D`
        kernel_initializer: initializer in `Conv2D`
        bias_initializer: initializer in `Conv2D`
        kernel_regularizer: regularizer in `Conv2D`
        use_gcnn: control use of gcnn
        conv_group: group determining gcnn operation
        depth_multiplier: Used to shrink the amount of parameters, used for fair Gconv/Conv comparison.
        name: name of the ops; will become `name + '_conv'`

    # Returns
        Convolution operation for `Conv2D`.
    """
    if use_gcnn and use_g_bn:
        return GBatchNorm(
            h=conv_group,
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs
        )

    return BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint, **kwargs)


def __Conv2D(filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             use_bias=True,
             kernel_initializer='he_normal',
             bias_initializer='zeros',
             kernel_regularizer=None,
             use_gcnn=None,
             conv_group=None,
             depth_multiplier=1,
             name=None):
    """Utility function to get conv operation, works with group to group
       convolution operations.

    # Arguments
        filters: filters in `Conv2D`
        kernel_size: height and width of the convolution kernel (tuple)
        strides: stride in 'Conv2D'
        padding: padding mode in `Conv2D`
        use_bias: bias mode in `Conv2D`
        kernel_initializer: initializer in `Conv2D`
        bias_initializer: initializer in `Conv2D`
        kernel_regularizer: regularizer in `Conv2D`
        use_gcnn: control use of gcnn
        conv_group: group determining gcnn operation
        depth_multiplier: Used to shrink the amount of parameters, used for fair Gconv/Conv comparison.
        name: name of the ops; will become `name + '_conv'`

    # Returns
        Convolution operation for `Conv2D`.
    """
    if use_gcnn:
        # Shrink the amount of filters used by GConv2D
        filters = int(round(filters * depth_multiplier))

        return GConv2D(
            filters, kernel_size,
            strides=strides,
            padding=padding,
            h_input=conv_group,
            h_output=conv_group,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            name=name.replace('conv', 'Gconv'))

    if depth_multiplier != 1:
        raise ValueError("Only use depth multiplier for gcnn networks.")

    return Conv2D(
        filters, kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name)


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    x = _preprocess_input(x, data_format=data_format)
    x *= 0.017  # scale values
    return x


def GDenseNet(mc_dropout, padding, nb_dense_block=3, growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
              bottleneck=False, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4, subsample_initial_block=False,
              include_top=True, weights=None, input_tensor=None, pooling=None, classes=10, activation='softmax',
              input_shape=None, depth=40, bn_momentum=0.99, use_gcnn=False, conv_group=None, depth_multiplier=1,
              use_g_bn=True, kernel_size=3, mc_bn=None):
    '''Instantiate the DenseNet architecture.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` dim ordering)
            or `(3, 224, 224)` (with `channels_first` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 8.
            E.g. `(224, 224, 3)` would be one valid value.
        depth: number or layers in the DenseNet
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. -1 indicates initial
            number of filters will default to 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
            Can be a -1, positive integer or a list.
            If -1, calculates nb_layer_per_block from the network depth.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be nb_dense_block
        bottleneck: flag to add bottleneck blocks in between dense blocks
        reduction: reduction factor of transition blocks.
            Note : reduction value is inverted to compute compression.
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling2D before the initial dense block.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization) or
            'imagenet' (pre-training on ImageNet)..
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        activation: Type of activation at the top layer. Can be one of
            'softmax' or 'sigmoid'. Note that if sigmoid is used,
             classes must be 1.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
            :param mc_bn:
            :param bn_momentum:
            :param padding:
            :param mc_dropout:
    '''

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_dense_net(classes, img_input, include_top, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn,
                           nb_layers_per_block, bottleneck, reduction, dropout_rate, weight_decay,
                           subsample_initial_block, pooling, activation, depth, nb_dense_block, growth_rate,
                           use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier,
                           kernel_size=kernel_size, nb_filter=nb_filter)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='densenet')

    return model


def GDenseNetFCN(input_shape, nb_dense_block=5, growth_rate=16, nb_layers_per_block=4, reduction=0.0, dropout_rate=0.0,
                 weight_decay=1E-4, init_conv_filters=48, include_top=True, weights=None, input_tensor=None, classes=1,
                 activation='softmax', upsampling_conv=128, upsampling_type='deconv', mc_dropout=False, padding='same',
                 bn_momentum=0.99, use_g_bn=True, use_gcnn=False, conv_group=None, mc_bn=None):
    '''Instantiate the DenseNet FCN architecture.
        Note that when using TensorFlow,
        for best performance you should set
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            nb_dense_block: number of dense blocks to add to end (generally = 3)
            growth_rate: number of filters to add per dense block
            nb_layers_per_block: number of layers in each dense block.
                Can be a positive integer or a list.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
            reduction: reduction factor of transition blocks.
                Note : reduction value is inverted to compute compression.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            init_conv_filters: number of layers in the initial convolution layer
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: one of `None` (random initialization) or
                'cifar10' (pre-training on CIFAR-10)..
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `channels_last` dim ordering)
                or `(3, 32, 32)` (with `channels_first` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
            activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.
            upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
            upsampling_type: Can be one of 'deconv', 'upsampling' and
                'subpixel'. Defines type of upsampling algorithm used.
            batchsize: Fixed batch size. This is a temporary requirement for
                computation of output shape in the case of Deconvolution2D layers.
                Parameter will be removed in next iteration of Keras, which infers
                output shape of deconvolution layers automatically.
        # Returns
            A Keras model instance.
            :param mc_bn:
    '''

    if weights not in {None}:
        raise ValueError('The `weights` argument should be '
                         '`None` (random initialization) as no '
                         'model weights are provided.')

    upsampling_type = upsampling_type.lower()

    if upsampling_type not in ['upsampling', 'deconv', 'subpixel']:
        raise ValueError('Parameter "upsampling_type" must be one of "upsampling", '
                         '"deconv" or "subpixel".')

    if input_shape is None:
        raise ValueError('For fully convolutional models, input shape must be supplied.')

    if type(nb_layers_per_block) is not list and nb_dense_block < 1:
        raise ValueError('Number of dense layers per block must be greater than 1. Argument '
                         'value was %d.' % (nb_layers_per_block))

    if activation not in ['softmax', 'sigmoid']:
        raise ValueError('activation must be one of "softmax" or "sigmoid"')

    if activation == 'sigmoid' and classes != 1:
        raise ValueError('sigmoid activation can only be used when classes = 1')

    # Determine proper input shape
    min_size = 2 ** nb_dense_block

    if K.image_data_format() == 'channels_first':
        if input_shape is not None:
            if ((input_shape[1] is not None and input_shape[1] < min_size) or
                    (input_shape[2] is not None and input_shape[2] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' + str(min_size) + ', got '
                                                                       '`input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (classes, None, None)
    else:
        if input_shape is not None:
            if ((input_shape[0] is not None and input_shape[0] < min_size) or
                    (input_shape[1] is not None and input_shape[1] < min_size)):
                raise ValueError('Input size must be at least ' +
                                 str(min_size) + 'x' + str(min_size) + ', got '
                                                                       '`input_shape=' + str(input_shape) + '`')
        else:
            input_shape = (None, None, classes)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_fcn_dense_net(classes, img_input, include_top, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn,
                               growth_rate, reduction, dropout_rate, weight_decay, nb_layers_per_block, upsampling_conv,
                               upsampling_type, init_conv_filters, input_shape, activation, conv_group, use_gcnn,
                               nb_dense_block)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='fcn-densenet')

    return model


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def __conv_block(ip, nb_filter, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, use_gcnn=None, conv_group=None, depth_multiplier=1, kernel_size=3,
                 block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and relu),
    and optionally a bottleneck layer.

    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming

     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        output tensor of block
        :param mc_bn:
        :param use_g_bn:
        :param bn_momentum:
        :param padding:
        :param mc_dropout:
    '''
    with K.name_scope('ConvBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                        name=name_or_none(block_prefix, '_bn'))(ip, training=mc_bn)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            x = __Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding=padding, use_bias=False,
                         kernel_regularizer=l2(weight_decay), name=name_or_none(block_prefix, '_bottleneck_conv2D'),
                         use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier)(x)
            x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                            name=name_or_none(block_prefix, '_bottleneck_bn'))(x, training=mc_bn)
            x = Activation('relu')(x)

        x = __Conv2D(nb_filter, (kernel_size, kernel_size), kernel_initializer='he_normal', padding=padding,
                     use_bias=False,
                     name=name_or_none(block_prefix, '_conv2D'), use_gcnn=use_gcnn, conv_group=conv_group,
                     depth_multiplier=depth_multiplier)(x)
        if dropout_rate:
            if mc_dropout:
                x = Dropout(dropout_rate)(x, training=True)
            else:
                x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, padding, mc_dropout, bn_momentum, growth_rate, use_g_bn, mc_bn,
                  return_concat_list=False, block_prefix=None, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  use_gcnn=None, conv_group=None, depth_multiplier=1, kernel_size=3, grow_nb_filters=True):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones

    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming

    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor

        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
        :param mc_bn:
        :param use_g_bn:
        :param bn_momentum:
        :param padding:
        :param mc_dropout:
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn, bottleneck=bottleneck,
                              dropout_rate=dropout_rate, weight_decay=weight_decay, use_gcnn=use_gcnn,
                              conv_group=conv_group, depth_multiplier=depth_multiplier, kernel_size=kernel_size,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([crop_to_fit(cb, x), cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_block(ip, nb_filter, padding, bn_momentum, use_g_bn, mc_bn, block_prefix=None, compression=1.0,
                       weight_decay=1e-4, use_gcnn=None, conv_group=None, depth_multiplier=1):
    '''
    Adds a pointwise convolution layer (with batch normalization and relu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.

    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.

    # Returns
        a keras tensor
        :param mc_bn:
        :param use_g_bn:
        :param bn_momentum:
        :param padding:
    '''
    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                        name=name_or_none(block_prefix, '_bn'))(ip, training=mc_bn)
        x = Activation('relu')(x)
        x = __Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding=padding,
                     use_bias=False, kernel_regularizer=l2(weight_decay), name=name_or_none(block_prefix, '_conv2D'),
                     use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        return x


def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4, block_prefix=None):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.

    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.

    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):

        if type == 'upsampling':
            x = UpSampling2D(name=name_or_none(block_prefix, '_upsampling'))(ip)
        elif type == 'subpixel':
            x = Conv2D(nb_filters, (3, 3), activation='relu', padding='valid', kernel_regularizer=l2(weight_decay),
                       use_bias=False, kernel_initializer='he_normal', name=name_or_none(block_prefix, '_conv2D'))(ip)
            x = SubPixelUpscaling(scale_factor=2, name=name_or_none(block_prefix, '_subpixel'))(x)
            x = Conv2D(nb_filters, (3, 3), activation='relu', padding='valid', kernel_regularizer=l2(weight_decay),
                       use_bias=False, kernel_initializer='he_normal', name=name_or_none(block_prefix, '_conv2D'))(x)
        else:
            x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='valid', strides=(2, 2),
                                kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
                                name=name_or_none(block_prefix, '_conv2DT'))(ip)
        return x


def __create_dense_net(nb_classes, img_input, include_top, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, pooling=None, activation='softmax', depth=40, nb_dense_block=3,
                       growth_rate=12, use_gcnn=False, conv_group=None, depth_multiplier=1, kernel_size=3,
                       nb_filter=-1):
    ''' Build the DenseNet model

    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling2D before the initial dense block.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.

    # Returns
        a keras tensor

    # Raises
        ValueError: in case of invalid argument for `reduction`
            or `nb_dense_block`
            :param mc_bn:
            :param use_g_bn:
            :param bn_momentum:
            :param padding:
            :param mc_dropout:
    '''
    with K.name_scope('DenseNet'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block):
                raise ValueError('If `nb_dense_block` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block
        print('nb_layers computed:', nb_layers, final_nb_layer)

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7)
            initial_strides = (2, 2)
        else:
            initial_kernel = (kernel_size, kernel_size)
            initial_strides = (1, 1)

        if use_gcnn:
            # Shrink the amount of parameters used during Gconv?
            # nb_filter = round(nb_filter * depth_multiplier)
            # Perhaps not do this for initial block... ^

            x = GConv2D(int(round(nb_filter * depth_multiplier)), initial_kernel, kernel_initializer='he_normal',
                        padding=padding, name='initial_Gconv2D',
                        strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay),
                        h_input='Z2', h_output=conv_group)(img_input)
        else:
            if depth_multiplier != 1:
                raise ValueError("Only use depth multiplier for gcnn networks.")

            x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding=padding,
                       name='initial_conv2D',
                       strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                            name='initial_bn')(x, training=mc_bn)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2), padding=padding)(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, padding, mc_dropout, bn_momentum,
                                         growth_rate, use_g_bn, mc_bn, block_prefix='dense_%i' % block_idx,
                                         bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay,
                                         use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier,
                                         kernel_size=kernel_size)
            # add transition_block
            x = __transition_block(x, nb_filter, padding, bn_momentum, use_g_bn, mc_bn,
                                   block_prefix='tr_%i' % block_idx, compression=compression, weight_decay=weight_decay,
                                   use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier)
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_block
        x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, padding, mc_dropout, bn_momentum, growth_rate,
                                     use_g_bn, mc_bn, block_prefix='dense_%i' % (nb_dense_block - 1),
                                     bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay,
                                     use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier,
                                     kernel_size=kernel_size)

        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                        name='final_bn')(x, training=mc_bn)
        x = Activation('relu')(x)

        if include_top:
            if use_gcnn:
                x = GroupPool(h_input=conv_group)(x)
            x = GlobalAveragePooling2D()(x)
            x = Dense(nb_classes, activation=activation)(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            if pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        return x


def __create_fcn_dense_net(nb_classes, img_input, include_top, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn,
                           growth_rate=12, reduction=0.0, dropout_rate=None, weight_decay=1e-4, nb_layers_per_block=4,
                           nb_upsampling_conv=128, upsampling_type='upsampling', init_conv_filters=48, input_shape=None,
                           activation='deconv', conv_group=None, use_gcnn=False, nb_dense_block=5):
    ''' Build the DenseNet-FCN model

    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        nb_layers_per_block: number of layers in each dense block.
            Can be a positive integer or a list.
            If positive integer, a set number of layers per dense block.
            If list, nb_layer is used as provided. Note that list size must
            be (nb_dense_block + 1)
        nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
        upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
            type of upsampling algorithm used.
        input_shape: Only used for shape inference in fully convolutional networks.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                    Note that if sigmoid is used, classes must be 1.

    # Returns
        a keras tensor

    # Raises
        ValueError: in case of invalid argument for `reduction`,
            `nb_dense_block` or `nb_upsampling_conv`.
            :param mc_bn:
    '''
    with K.name_scope('DenseNetFCN'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if concat_axis == 1:  # channels_first dim ordering
            _, rows, cols = input_shape
        else:
            rows, cols, _ = input_shape

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # check if upsampling_conv has minimum number of filters
        # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
        if not (nb_upsampling_conv > 12 and nb_upsampling_conv % 4 == 0):
            raise ValueError('Parameter `nb_upsampling_conv` number of channels must '
                             'be a positive number divisible by 4 and greater than 12')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block + 1):
                raise ValueError('If `nb_dense_block` is a list, its length must be '
                                 '(`nb_dense_block` + 1)')

            bottleneck_nb_layers = nb_layers[-1]
            rev_layers = nb_layers[::-1]
            nb_layers.extend(rev_layers[1:])
        else:
            bottleneck_nb_layers = nb_layers_per_block
            nb_layers = [nb_layers_per_block] * (2 * nb_dense_block + 1)

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        x = Conv2D(init_conv_filters, (7, 7), kernel_initializer='he_normal', padding='valid', name='initial_conv2D',
                   use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)
        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, epsilon=1.1e-5, axis=concat_axis, name='initial_bn')(x,
                                                                                                             training=mc_bn)
        x = Activation('relu')(x)

        nb_filter = init_conv_filters

        skip_list = []

        # Add dense blocks and transition down block
        for block_idx in range(nb_dense_block):
            x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, padding, mc_dropout, bn_momentum,
                                         growth_rate, use_g_bn, mc_bn, block_prefix='dense_%i' % block_idx,
                                         dropout_rate=dropout_rate, weight_decay=weight_decay)

            # Skip connection
            skip_list.append(x)

            # add transition_block
            x = __transition_block(x, nb_filter, padding, bn_momentum, use_g_bn, mc_bn,
                                   block_prefix='tr_%i' % block_idx, compression=compression, weight_decay=weight_decay)

            nb_filter = int(nb_filter * compression)  # this is calculated inside transition_down_block

        # The last dense_block does not have a transition_down_block
        # return the concatenated feature maps without the concatenation of the input
        _, nb_filter, concat_list = __dense_block(x, bottleneck_nb_layers, nb_filter, padding, mc_dropout, bn_momentum,
                                                  growth_rate, use_g_bn, mc_bn, return_concat_list=True,
                                                  block_prefix='dense_%i' % nb_dense_block, dropout_rate=dropout_rate,
                                                  weight_decay=weight_decay)

        skip_list = skip_list[::-1]  # reverse the skip list

        # Add dense blocks and transition up block
        for block_idx in range(nb_dense_block):
            n_filters_keep = growth_rate * nb_layers[nb_dense_block + block_idx]

            # upsampling block must upsample only the feature maps (concat_list[1:]),
            # not the concatenation of the input with the feature maps (concat_list[0].
            l = concatenate(concat_list[1:], axis=concat_axis)

            t = __transition_up_block(l, nb_filters=n_filters_keep, type=upsampling_type, weight_decay=weight_decay,
                                      block_prefix='tr_up_%i' % block_idx)

            # concatenate the skip connection with the transition block
            x = concatenate([t, skip_list[block_idx]], axis=concat_axis)

            # Dont allow the feature map size to grow in upsampling dense blocks
            x_up, nb_filter, concat_list = __dense_block(x, nb_layers[nb_dense_block + block_idx + 1],
                                                         nb_filter=growth_rate, padding=padding, mc_dropout=mc_dropout,
                                                         bn_momentum=bn_momentum, growth_rate=growth_rate,
                                                         use_g_bn=use_g_bn, mc_bn=mc_bn, return_concat_list=True,
                                                         block_prefix='dense_%i' % (nb_dense_block + 1 + block_idx),
                                                         dropout_rate=dropout_rate, weight_decay=weight_decay,
                                                         grow_nb_filters=False)

        if include_top:
            x = Conv2D(nb_classes, (1, 1), activation='linear', padding='valid', use_bias=False)(x_up)

            if K.image_data_format() == 'channels_first':
                channel, row, col = input_shape
            else:
                row, col, channel = input_shape

            x = Reshape((row * col, nb_classes))(x)
            x = Activation(activation)(x)
            x = Reshape((row, col, nb_classes))(x)
        else:
            x = x_up

        return x
