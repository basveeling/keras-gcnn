from keras import backend as K
from keras.engine import Layer
from keras.utils import get_custom_objects


class GroupPool(Layer):
    def __init__(self, h_input, **kwargs):
        super(GroupPool, self).__init__(**kwargs)
        self.h_input = h_input

    def build(self, input_shape):
        self.shape = input_shape
        super(GroupPool, self).build(input_shape)

    @property
    def nti(self):
        nti = 1
        if self.h_input == 'C4':
            nti *= 4
        elif self.h_input == 'D4':
            nti *= 8
        return nti

    def call(self, x):
        shape = K.shape(x)
        stack_shape = K.stack([shape[0], shape[1], shape[2], shape[3] // self.nti, self.nti])
        input_reshaped = K.reshape(x, stack_shape)
        mean_per_group = K.mean(input_reshaped, -1)
        return mean_per_group

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[3] // self.nti)

    def get_config(self):
        config = super(GroupPool, self).get_config()
        config['h_input'] = self.h_input
        return config


get_custom_objects().update({'GroupPool': GroupPool})
