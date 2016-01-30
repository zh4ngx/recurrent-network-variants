from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

from feedback_cell import GFCell


class VanillaRNNCell(rnn_cell.RNNCell):
    """The most basic RNN cell."""

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

    def __call__(self, inputs, state, scope=None):
        """
         Recurrence functionality here
         In contrast to tensorflow implementation, variables will be more explicit
         :param inputs: 2D Tensor with shape [batch_size x self.input_size]
         :param state: 2D Tensor with shape [batch_size x self.state_size]
         :param scope: VariableScope for the created subgraph; defaults to class name
         :return:
             h_t - Output: A 2D Tensor with shape [batch_size x self.output_size]
             h_t - New state: A 2D Tensor with shape [batch_size x self.state_size].
             (the new state is also the output in a vanilal RNN cell)
         """
        with tf.variable_scope(scope or type(self).__name__):
            x = inputs
            h_t_1 = state
            W = tf.get_variable("W", [self.input_size, self.state_size],
                                initializer=tf.random_uniform_initializer(-0.1, 0.1))
            U = tf.get_variable("U", [self.state_size, self.state_size],
                                initializer=tf.random_uniform_initializer(-0.1, 0.1))
            b = tf.get_variable("b", [self.state_size], tf.constant_initializer(0.0))
            h_t = tf.tanh(tf.matmul(x, W) + tf.matmul(h_t_1, U) + b)

        return h_t, h_t


class GRUCell(rnn_cell.RNNCell):
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

    @property
    def __call__(self, inputs, state, scope=None):
        """
        Recurrence functionality here
        In contrast to tensorflow implementation, variables will be more explicit
        :param inputs: 2D Tensor with shape [batch_size x self.input_size]
        :param state: 2D Tensor with shape [batch_size x self.state_size]
        :param scope: VariableScope for the created subgraph; defaults to class name
        :return:
            h_t - Output: A 2D Tensor with shape [batch_size x self.output_size]
            h_t - New state: A 2D Tensor with shape [batch_size x self.state_size].
            (the new state is also the output in a GRU cell)
        """
        with tf.variable_scope(scope or type(self).__name__):
            h_t_prev, _ = tf.split(1, 2, state)
            x_t = inputs
            with tf.variable_scope("Update Gate"):
                W_z = tf.get_variable("W_z", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                U_z = tf.get_variable("U_z", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                b_z = tf.get_variable("b_z", [self._num_units], tf.constant_initializer(0.0))

                z_t = tf.sigmoid(tf.matmul(x_t, W_z) + tf.matmul(h_t_prev, U_z) + b_z, name="z_t")

            with tf.variable_scope("Reset Gate"):
                W_r = tf.get_variable("W_r", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                U_r = tf.get_variable("U_r", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                b_r = tf.get_variable("b_r", [self._num_units], tf.constant_initializer(1.0))

                r_t = tf.sigmoid(tf.matmul(x_t, W_r) + tf.matmul(h_t_prev, U_r) + b_r, name="r_t")

            with tf.variable_scope("Candidate"):
                # New memory content
                W = tf.get_variable("W", [self.input_size, self._num_units],
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                U = tf.get_variable("U", [self.input_size, self._num_units],
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                b = tf.get_variable("b", [self._num_units], tf.constant_initializer(0.0))
                hc_t = tf.tanh(tf.matmul(x_t, W) + tf.mul(r_t, tf.matmul(h_t_prev, U) + b))

            with tf.Variable("Output"):
                h_t = tf.mul(z_t, hc_t) + tf.mul((1 - z_t), h_t_prev)

        return h_t, h_t


class GFGRUCell(GFCell):
    def __call__(self, inputs, state, full_state, layer_sizes, scope=None):
        """
        Recurrence functionality here
        In contrast to tensorflow implementation, variables will be more explicit
        :param inputs: 2D Tensor with shape [batch_size x self.input_size]
        :param state: 2D Tensor with shape [batch_size x self.state_size]
        :param full_state: 2D Tensor with shape [batch_size x self.full_state_size]
        :param scope: VariableScope for the created subgraph; defaults to class name
        :return:
            h_t - Output: A 2D Tensor with shape [batch_size x self.output_size]
            h_t - New state: A 2D Tensor with shape [batch_size x self.state_size].
            (the new state is also the output in a GRU cell)
        """
        with tf.variable_scope(scope or type(self).__name__):
            h_t_prev, _ = tf.split(1, 2, state)
            x_t = inputs
            with tf.variable_scope("Update Gate"):
                W_z = tf.get_variable("W_z", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                U_z = tf.get_variable("U_z", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                b_z = tf.get_variable("b_z", [self._num_units], tf.constant_initializer(0.0))

                z_t = tf.sigmoid(tf.matmul(x_t, W_z) + tf.matmul(h_t_prev, U_z) + b_z, name="z_t")

            with tf.variable_scope("Reset Gate"):
                W_r = tf.get_variable("W_r", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                U_r = tf.get_variable("U_r", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                b_r = tf.get_variable("b_r", [self._num_units], tf.constant_initializer(1.0))

                r_t = tf.sigmoid(tf.matmul(x_t, W_r) + tf.matmul(h_t_prev, U_r) + b_r, name="r_t")

            with tf.variable_scope("Candidate"):
                # New memory content
                W = tf.get_variable("W", [self.input_size, self._num_units],
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))

                b = tf.get_variable("b", [self._num_units], tf.constant_initializer(0.0))

                summation_term = self.compute_feedback(x_t, full_state, layer_sizes)
                hc_t = tf.tanh(tf.matmul(x_t, W) + tf.mul(r_t, summation_term))

            with tf.Variable("Output"):
                h_t = tf.mul(z_t, hc_t) + tf.mul((1 - z_t), h_t_prev)

        return h_t, h_t
