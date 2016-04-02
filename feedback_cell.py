from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import array_ops


class GFCell(object):
    """Abstract object representing a cell in a Gated Feedback RNN
    Operates like an RNNCell
    """

    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        """Return state tensor (shape [batch_size x state_size]) filled with 0.

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          A 2D Tensor of shape [batch_size x state_size] filled with zeros.
        """
        zeros = array_ops.zeros(
                array_ops.pack([batch_size, self.state_size]), dtype=dtype)
        zeros.set_shape([None, self.state_size])
        return zeros

    def __call__(self, inputs, state, full_state, layer_sizes, scope=None):
        raise NotImplementedError("Abstract method")

    def compute_feedback(self, inputs, full_state, layer_sizes, scope=None):
        with tf.variable_scope("Global Reset"):
            cur_state_pos = 0
            full_state_size = sum(layer_sizes)
            summation_term = tf.get_variable("summation", self.state_size, initializer=tf.constant_initializer())
            for i, layer_size in enumerate(layer_sizes):
                with tf.variable_scope("Cell%d" % i):
                    # Compute global reset gate
                    w_g = tf.get_variable("w_g", self.input_size, initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    u_g = tf.get_variable("u_g", full_state_size, initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    g__i_j = tf.sigmoid(tf.matmul(inputs, w_g) + tf.matmul(full_state, u_g))

                    # Accumulate sum
                    h_t_1 = \
                        tf.slice(
                                full_state,
                                [0, cur_state_pos],
                                [-1, layer_size]
                        )
                    cur_state_pos += layer_size
                    U = tf.get_variable("U", [self.input_size, self._num_units],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))
                    b = tf.get_variable("b", self.state_size, initializer=tf.constant_initializer(1.))
                    summation_term = tf.add(summation_term, g__i_j * tf.matmul(U, h_t_1) + b)

        return summation_term


class FeedbackCell(object):
    """
    MultiRNNCell composed of stacked cells that interact across layers
    Based on http://arxiv.org/pdf/1502.02367v4.pdf
    """

    def __init__(self, num_units, cells):
        self._num_units = num_units
        self._cells = cells
        for cell in cells:
            if not isinstance(cell, GFCell):
                raise ValueError("Cells must be of type GFCell")

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def state_size(self):
        return self._num_units

    def __call__(self, inputs, hs_prev, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Conveniently the concatenation of all hidden states at t-1
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            new_hs = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("Cell%d" % i):
                    cur_state = array_ops.slice(
                            state, [0, cur_state_pos], [-1, cell.state_size])
                    cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, cur_state, hs_prev, self.state_size)
                    new_states.append(new_state)
                    new_hs.append(cur_inp)

        return cur_inp, new_hs, array_ops.concat(1, new_states)
