import tensorflow as tf

from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D, Conv2D

from tensorflow.keras import layers

# Symmetrical layer
#@keras_export('keras.layers.Conv2DSymmetrical', 'keras.layers.Convolution2DSymmetrical')
class Conv2DSymm(layers.Layer):
    """2D convolution layer (e.g. spatial convolution over images)."""

    def __init__(self,
                 filters,  # NOTE: Output size is 2x the number of filters.
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv2DSymm, self).__init__(**kwargs)
        # Create a new conv2d layer with the given parameters.
        self.convL = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        #self.filters = filters

    # Apply operation T to matrix.
    # This is both 'flip' and 'shuffle'
    def trans(self, X):
        flt = X.shape[-1] // 2
        a = X[:, :, :, 0:flt]
        b = X[:, :, :, flt:]
        res = self.flip(tf.concat([b,a], axis=-1))
        print("trannsssssssssss", a.shape, b.shape, res.shape, flush=True)
        return res

    # reverse x along symmetry. TODO: Check axis.
    def flip(self, X):
        print("FLIP", X.shape, flush=True)
        return tf.reverse(X, axis=[-2]) # When it's ?,8,4,1, it's tall and thin... Right?

    def call(self, inputs):

        # f(X) = { f'(X)       }
        #        { f'(TX) R^-1 }

        # Just call with no rotation, then flipped.
        a1 = self.convL(inputs)
        # Transformed input 2
        inp2 = self.trans(inputs)
        a2 = self.flip(self.convL(inp2))  # Since flip is it's own inverse.

        return tf.concat([a1,a2], axis=-1)

    def compute_output_shape(self, input_shape):
        # It's just 4 times THICCer than conv2d would be?
        sh = self.convL.compute_output_shape(input_shape)
        sh[-1] *= 2
        return sh


# Taken from:
# https://github.com/danielegrattarola/spektral/blob/ef83b93fa8d0b8015a0793371bb63b8447398b42/spektral/layers/convolutional.py
class GraphConv(Layer):
    """
    A graph convolutional layer as presented by
    [Kipf & Welling (2016)](https://arxiv.org/abs/1609.02907).
    **Mode**: single, mixed, batch.

    This layer computes:
    $$
        Z = \\sigma( \\tilde{A} XW + b)
    $$
    where \(X\) is the node features matrix, \(\\tilde{A}\) is the normalized
    Laplacian, \(W\) is the convolution kernel, \(b\) is a bias vector, and
    \(\\sigma\) is the activation function.

    **Input**

    - Node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - Normalized Laplacian of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension); see `spektral.utils.convolution.localpooling_filter`.

    **Output**

    - Node features with the same shape of the input, but the last dimension
    changed to `channels`.

    **Arguments**

    - `channels`: integer, number of output channels;
    - `activation`: activation function to use;
    - `use_bias`: whether to add a bias to the linear transformation;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.

    **Usage**

    ```py
    # Load data
    A, X, _, _, _, _, _, _ = citation.load_data('cora')
    # Preprocessing operations
    fltr = utils.localpooling_filter(A)
    # Model definition
    X_in = Input(shape=(F, ))
    fltr_in = Input((N, ), sparse=True)
    output = GraphConv(channels)([X_in, fltr_in])
    ```
    """
    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super().__init__(**kwargs)
            self.channels = channels
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel)
        output = filter_dot(fltr, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

# As before but always the same adj matrix for the graph.
class ConstGraphConv(Layer):
    def __init__(self,
                 channels,
                 fltr,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
            super().__init__(**kwargs)
            self.channels = channels
            self.fltr = fltr
            self.activation = activations.get(activation)
            self.use_bias = use_bias
            self.kernel_initializer = initializers.get(kernel_initializer)
            self.bias_initializer = initializers.get(bias_initializer)
            self.kernel_regularizer = regularizers.get(kernel_regularizer)
            self.bias_regularizer = regularizers.get(bias_regularizer)
            self.activity_regularizer = regularizers.get(activity_regularizer)
            self.kernel_constraint = constraints.get(kernel_constraint)
            self.bias_constraint = constraints.get(bias_constraint)
            self.supports_masking = False

    def build(self, input_shape):
        # assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
            self.built = True

    def call(self, features):
        # features = inputs
        # fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel)
        output = filter_dot(self.fltr, output)
        print("heyy", self.fltr, output)
        # output = K.batch_dot(self.fltr, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (self.channels,)
        print(input_shape, self.channels)

        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

def filter_dot(fltr, features):
    """
    Performs the multiplication of a graph filter (N x N) with the node features,
    automatically dealing with single, mixed, and batch modes.
    :param fltr: the graph filter(s) (N x N in single and mixed mode,
    batch x N x N in batch mode).
    :param features: the node features (N x F in single mode, batch x N x F in
    mixed and batch mode).
    :return: the filtered features.
    """
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        if len(K.int_shape(fltr)) == 3:
            # Batch mode
            return K.batch_dot(fltr, features)
        else:
            # Mixed mode
            return mixed_mode_dot(fltr, features)

def mixed_mode_dot(A, B):
    """
    Computes the equivalent of `tf.einsum('ij,bjk->bik', fltr, output)`, but
    works for both dense and sparse input filters.
    :param A: rank 2 Tensor or SparseTensor.
    :param B: rank 3 Tensor or SparseTensor.
    :return: rank 3 Tensor or SparseTensor.
    """
    s_0_, s_1_, s_2_ = K.int_shape(B)
    B_T = transpose(B, (1, 2, 0))
    B_T = reshape(B_T, (s_1_, -1))
    output = single_mode_dot(A, B_T)
    output = reshape(output, (s_1_, s_2_, -1))
    output = transpose(output, (2, 0, 1))

    return output

def transpose(A, perm=None, name=None):
    if K.is_sparse(A):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(A, perm=perm, name=name)

def reshape(A, shape=None, name=None):
    """
    Reshapes A according to shape, dealing with sparse A automatically.
    :param A: Tensor or SparseTensor.
    :param shape: new shape.
    :param name: name for the operation.
    :return: Tensor or SparseTensor.
    """
    if K.is_sparse(A):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(A, shape=shape, name=name)


def single_mode_dot(A, B):
    """
    Dot product between two rank 2 matrices. Deals automatically with either A
    or B being sparse.
    :param A: rank 2 Tensor or SparseTensor.
    :param B: rank 2 Tensor or SparseTensor.
    :return: rank 2 Tensor or SparseTensor.
    """
    a_sparse = K.is_sparse(A)
    b_sparse = K.is_sparse(B)
    if a_sparse and b_sparse:
        raise ValueError('Sparse x Sparse matmul is not implemented yet.')
    elif a_sparse:
        output = tf.sparse_tensor_dense_matmul(A, B)
    elif b_sparse:
        output = transpose(
            tf.sparse_tensor_dense_matmul(
                transpose(B), transpose(A)
            )
        )
    else:
        output = tf.matmul(A, B)

    return output
