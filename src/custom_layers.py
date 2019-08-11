import tensorflow.keras as tfk


class InstanceNorm(tfk.layers.Layer):
    """Instance normalization layer.

    Normalizes the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    of each feature map for each instance in batch close to 0 and the standard
    deviation close to 1.

    Args:
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv1D` layer with
            `data_format="channels_last"`,
            set `axis=-1` in `InstanceNorm`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        Input shape: Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a Sequential model.
        Output shape: Same shape as input.

    References:
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """

    def __init__(self,
                 axis=-1,
                 epsilon=1e-3,
                 mean=0,
                 stddev=1,
                 **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.mean = mean
        self.stddev = stddev
        super(InstanceNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = tfk.layers.InputSpec(ndim=ndim)
        super(InstanceNorm, self).build(input_shape)

    def call(self, inputs):
        input_shape = tfk.backend.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = tfk.backend.mean(inputs, reduction_axes, keepdims=True)
        stddev = tfk.backend.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean + self.mean) / stddev * self.stddev
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(InstanceNorm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))