import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.constraints import non_neg
from keras.initializers import constant
from keras.layers import InputSpec, Dropout


class CosineSim(Layer):
    """ Computes the cosine similarity between 2 tensors,
    where the first one is 2D and the second one 3D"""

    def __init__(self, **kwargs):
        super(CosineSim, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CosineSim, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x

        q_norm = K.l2_normalize(K.expand_dims(a, 1), 2)
        l_norm = K.l2_normalize(b, 2)
        output = K.sum(q_norm * l_norm, axis=2)

        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_b[1])


class Linear(Layer):
    """ Linear layer in order to weigh each modality

    # Example

    # Arguments
        n: integer, repetition factor.
    # Input shape
        2D tensor of shape `(num_samples, features)`.
    # Output shape
        3D tensor of shape `(num_samples, n, features)`.

    """

    def __init__(self, **kwargs):
        super(Linear, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weights variable for this layer.
        self.kernel1 = self.add_weight(name="modality_weight_1",
                                       shape=(1,),
                                       initializer=constant(value=0.0),
                                       trainable=True,
                                       constraint=non_neg())

        super(Linear, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return (a * self.kernel1) + (b * (1 - self.kernel1))  # self.kernel2

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a


class WordDropout(Layer):
    """Taken from:
    https://github.com/aravindsiv/dan_qa/blob/master/custom_layers.py"""

    def __init__(self, rate, **kwargs):
        super(WordDropout, self).__init__()
        self.rate = min(1., max(0., rate))
        self.supports_masking = True

    def build(self, input_shape):
        super(WordDropout, self).build(input_shape)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.0:
            def dropped_inputs():
                input_shape = K.shape(inputs)
                batch_size = input_shape[0]
                n_time_steps = input_shape[1]
                mask = tf.random_uniform((batch_size, n_time_steps, 1)) >= self.rate
                w_drop = K.cast(mask, "float32") * inputs
                return w_drop

            return K.in_train_phase(dropped_inputs, inputs, training=training)

        return inputs

    def get_config(self):
        config = {"rate": self.rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosineSim3D(Layer):
    """
      A [batch x n x d] tensor of n rows with d dimensions
      B [batch x m x d] tensor of n rows with d dimensions

      returns:
      D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    def __init__(self, **kwargs):
        super(CosineSim3D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CosineSim3D, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)

        def l2_norm(x, axis=None):
            """
            :param x: tensor
            :param axis: axis
            :return: the l2 norm along specified axis
            """
            square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
            norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

            return norm

        a, b = x

        A_mag = l2_norm(a, axis=2)  # tf.norm(tensor=a,axis=2,keepdims=True)#K.l2_normalize(x=a,axis=2)
        B_mag = l2_norm(b, axis=2)  # tf.norm(tensor=b,axis=2,keepdims=True)#K.l2_normalize(x=b,axis=2)
        num = K.batch_dot(a, K.permute_dimensions(b, (0, 2, 1)))
        den = (A_mag * K.permute_dimensions(B_mag, (0, 2, 1)))
        output = num / den

        # better remove them from this layer
        output = K.sum(output, axis=2)
        output = K.softmax(output, axis=-1)
        output = K.expand_dims(output, axis=-1)
        output = K.tile(output, (1, 1, 300))

        return output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return (shape_a[0], shape_a[1], 300)


class KeywordScores(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(KeywordScores, self).__init__(**kwargs)

    def build(self, input_shape):
        super(KeywordScores, self).build(input_shape)

    def call(self, x):
        output = K.sum(x, axis=2)
        output = K.softmax(output, axis=-1)
        output = K.expand_dims(output, axis=-1)
        output = K.tile(output, (1, 1, self.output_dim))

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class AverageWords(Layer):
    """Taken from:
      https://github.com/aravindsiv/dan_qa/blob/master/custom_layers.py"""

    def __init__(self, **kwargs):
        super(AverageWords, self).__init__()
        self.supports_masking = True

    def call(self, x, mask=None):
        axis = K.ndim(x) - 2
        if mask is not None:
            summed = K.sum(x, axis=axis)
            n_words = K.expand_dims(K.sum(K.cast(mask, "float32"), axis=axis), axis)
            return summed / n_words
        else:
            return K.mean(x, axis=axis)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        dimensions = list(input_shape)
        n_dimensions = len(input_shape)
        del dimensions[n_dimensions - 2]
        return tuple(dimensions)


class TimestepDropout(Dropout):
    """Timestep Dropout.

    This version performs the same function as Dropout, however it drops
    entire timesteps (e.g., words embeddings in NLP tasks) instead of individual elements (features).

    # Arguments
        rate: float between 0 and 1. Fraction of the timesteps to drop.

    # Input shape
        3D tensor with shape:
        `(samples, timesteps, channels)`

    # Output shape
        Same as input

    # References
        - A Theoretically Grounded Application of Dropout in Recurrent Neural Networks (https://arxiv.org/pdf/1512.05287)
    """

    def __init__(self, rate, **kwargs):
        super(TimestepDropout, self).__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = K.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)
        return noise_shape