import tensorflow as tf
import numpy as np

from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D, Conv2D

from tensorflow.keras import layers

from collections import namedtuple

# Class defining symmetry groups as a graph.

# Definition for a generator of a finite group.
FiniteGroupGen = namedtuple('FiniteGroup', 'operation, period, mapping')

def no_op(X, times):
    return X
IdentityGroup = [FiniteGroupGen(no_op, 1, [0])]

# Horizontal symmetry group.
def flipH(X, times=1):
    return X if (times % 2 == 0) else tf.reverse(X, axis=[-2])
HorizontalSymGen = [FiniteGroupGen(flipH, 2, [1,0])]

# Vertical symmetry group.
def flipV(X, times=1):
    return X if (times % 2 == 0) else tf.reverse(X, axis=[-3])
VerticalSymGen = [FiniteGroupGen(flipV, 2, [1,0])]

FlipsSymGen = [FiniteGroupGen(flipH, 2, [2,3, 0,1]),
               FiniteGroupGen(flipV, 2, [1,0, 3,2])]

# Rotation symmetry group.
def rot(X, times=1):
    return tf.image.rot90(X, times)

RotationSymGen = [FiniteGroupGen(rot, 4, [1, 2, 3, 0])]

# Generator for the full S_8 symmetry/rotation group.
RotFlipSymGen = [FiniteGroupGen(flipH, 2, [4,7,6,5, 0,3,2,1]),
                 FiniteGroupGen(rot,   4, [1,2,3,0, 5,6,7,4])]
# Symmetrical layer
class Conv2DSym(layers.Layer):
    """2D convolution layer (e.g. spatial convolution over images)."""

    def __init__(self,
                 filters,  # NOTE: Output 'thickness' is groupsize * the number of filters.
                 kernel_size,
                 symmetry_group,
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
        super(Conv2DSym, self).__init__(**kwargs)
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

        self.generators = generators
        self.num_elem = int(np.prod([g.period for g in generators]))

    # Apply operation T to matrix.
    # This is both 'flip/rotate' and 'shuffle'
    def _applyT(self, X, fg, times=1):
        # Split tensor into a list of pieces
        # - one for each element in the group.
        parts = tf.split(X, num_or_size_splits=self.num_elem, axis=-1)

        # Reorder parts according to map (times times).
        for i in range(times % fg.period):
            parts = [parts[i] for i in fg.mapping]

        # Concatinate and apply group operation
        X2 = self._applyG(tf.concat(parts, axis=-1), fg, times)
        #print("_applyTx2", X2.shape[-1])
        return X2

    # X is input tensor, fg is finite group.
    def _applyG(self, X, fg, times=1):
        # Apply group op times number of .. times.
        return fg.operation(X, times)

    # Takes conv layer and target tensor.
    def call(self, inputs):
        res = []
        # Enumerate the generators.
        if len(self.generators) == 1:
            # Go through generator's 'loop'.
            fg = self.generators[0]
            for i in range(fg.period):

                # f(X) = { f'(X)         }
                #        { f'(TX)   G^-1 }
                #        { f'(T^2X) G^-2 }
                #        {      ...      }

                a = self._applyT(inputs, fg, i)
                a = self.convL(a)
                a = self._applyG(a, fg, -i)
                res.append(a)

        elif len(self.generators) == 2:
            fg1, fg2 = self.generators[0], self.generators[1]

            for i in range(fg1.period):
                for j in range(fg2.period):
                    # Apply fg2 then fg1.
                    a = self._applyT(self._applyT(inputs, fg2, j), fg1, i)
                    a = self.convL(a)
                    # Undo fg1 then fg2
                    a = self._applyG(self._applyG(a, fg1, -i), fg2, -j)
                    res.append(a)

        else:
            # Can't handle more than 2 generators yet...
            raise NotImplementedError


        return tf.concat(res, axis=-1)

    def compute_output_shape(self, input_shape):
        # It's just 4 times THICCer than conv2d would be?
        sh = self.convL.compute_output_shape(input_shape)
        sh[-1] *= self.num_elem
        return sh

    def get_config(self):
        super().get_config()

# Concat layer
class ConcatSym(layers.Layer):

    def __init__(self, generators):
        super(ConcatSym, self).__init__()
        # TODO: Remove this line from every layer somehow.
        self.num_elem = int(np.prod([g.period for g in generators]))

    def call(self, inputs):
        #print("ConcatSym input:", inputs[0].shape, inputs[1].shape, flush=True)

        # List of lists of pieces.
        pieces = [tf.split(x, num_or_size_splits=self.num_elem, axis=-1) for x in inputs]
        # Interpolate (intercalate?) by transposing then flattening.
        pieces = np.array(pieces).T.flatten().tolist()
        # Concat into single tensor
        return tf.concat(pieces, axis=-1)

    #def compute_output_shape(self, input_shape):
    #    # It's just the sum of the inputs...
    #    return ???


class Lift(layers.Layer):
    def __init__(self, generators):
        super(Lift, self).__init__()
        self.num_elem = int(np.prod([g.period for g in generators]))
        print("lft", self.num_elem, flush=True)

    def call(self, inputs):
        # Just repeat input num_elem times
        xs = [inputs for _ in range(self.num_elem)]
        return tf.concat(xs, axis=-1)

    def compute_output_shape(self, input_shape):
        input_shape[-1]*=self.num_elem
        return input_shape


class Drop(layers.Layer):
    def __init__(self, generators):
        super(Drop, self).__init__()
        self.num_elem = int(np.prod([g.period for g in generators]))

    def call(self, inputs):
        # Split into parts, and simply sum.
        parts = tf.split(inputs, num_or_size_splits=self.num_elem, axis=-1)

        # Add all parts...
        #ans = tf.add_n(parts) #/ self.num_elem

        # Take max values of all the parts.
        ans = parts.pop(0)
        for p in parts:
            ans = tf.maximum(ans, p)

        return ans

    def compute_output_shape(self, input_shape):
        input_shape[-1] //= self.num_elem
        return input_shape
