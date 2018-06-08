from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util
from keras import backend as K
from keras.engine import InputSpec
from keras.layers import BatchNormalization
from keras.utils import get_custom_objects


class GBatchNorm(BatchNormalization):

    def __init__(self, h, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True, beta_initializer='zeros',
                 gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        self.h = h
        if axis != -1:
            raise ValueError('Assumes 2D input with channels as last dimension.')
        super(GBatchNorm, self).__init__(axis=axis, momentum=momentum, epsilon=epsilon, center=center, scale=scale,
                                         beta_initializer=beta_initializer, gamma_initializer=gamma_initializer,
                                         moving_mean_initializer=moving_mean_initializer,
                                         moving_variance_initializer=moving_variance_initializer,
                                         beta_regularizer=beta_regularizer, gamma_regularizer=gamma_regularizer,
                                         beta_constraint=beta_constraint, gamma_constraint=gamma_constraint, **kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        self.gconv_indices, self.gconv_shape_info, w_shape = gconv2d_util(h_input=self.h, h_output=self.h,
                                                                          in_channels=input_shape[-1],
                                                                          out_channels=input_shape[-1],
                                                                          ksize=1)
        if self.h == 'C4':
            dim //= 4
        elif self.h == 'D4':
            dim //= 8
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        def repeat(w):
            n = 1
            if self.h == 'C4':
                n *= 4
            elif self.h == 'D4':
                n *= 8
            elif self.h == 'Z2':
                n *= 1
            else:
                raise ValueError('Wrong h: %s' % self.h)

            return K.reshape(
                K.tile(
                    K.expand_dims(w, -1), [1, n]), [-1])

        self.repeated_gamma = repeat(self.gamma)
        self.repeated_beta = repeat(self.beta)

        self.repeated_moving_mean = repeat(self.moving_mean)
        self.repeated_moving_variance = repeat(self.moving_variance)
        self.built = True

    def call(self, inputs, training=None):

        def unrepeat(w):
            n = 1
            if self.h == 'C4':
                n *= 4
            elif self.h == 'D4':
                n *= 8
            elif self.h == 'Z2':
                n *= 1
            else:
                raise ValueError('Wrong h: %s' % self.h)

            return K.mean(
                K.reshape(w, (K.int_shape(w)[0] // n, n)), -1)

        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.repeated_moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.repeated_moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.repeated_beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.repeated_gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.repeated_moving_mean,
                    self.repeated_moving_variance,
                    self.repeated_beta,
                    self.repeated_gamma,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.repeated_gamma, self.repeated_beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 unrepeat(mean),
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 unrepeat(variance),
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        return dict(list({'h': self.h}.items()) +
                    list(super(GBatchNorm, self).get_config().items()))


get_custom_objects().update({'GBatchNorm': GBatchNorm})
