from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class LSTMNCell(object):
    def __init__(self, num_units, memory_capacity):
        self._num_units = num_units
        self._memory_capacity = memory_capacity

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units, self._memory_capacity


    def __call__(self, inputs, memory_tape, current_timestep, scope=None):
        """
        Recurrence functionality here
        In contrast to tensorflow implementation, variables will be more explicit
        :param inputs: 2D Tensor with shape [batch_size x self.input_size]
        :param state: 2D Tensor with shape [batch_size x self.state_size x memory_capacity]
        :param scope: VariableScope for the created subgraph; defaults to class name
        :return:
            h_t - Output: A 2D Tensor with shape [batch_size x self.output_size]
            h_t - New state: A 2D Tensor with shape [batch_size x self.state_size].
            (the new state is also the output in a GRU cell)
        """
        with tf.variable_scope(scope or type(self).__name__):
            c, h = tf.split(1, 2, memory_tape)
            x_t = inputs

            with tf.variable_scope("attention"):
                v = tf.get_variable("v", [self.output_size],
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                W_h = tf.get_variable("W_h", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                W_x = tf.get_variable("W_x", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                W_h_c = tf.get_variable("W_h_c", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                for i in range(0, current_timestep):
                    with tf.variable_scope(i):
                        a = tf.matmul(
                            v,
                            tf.tanh(
                                tf.matmul(W_h, memory_tape[:, :, i]) +
                                tf.matmul(W_x, x_t) +
                                tf.matmul(W_h_c, memory_tape[:, :, i]) # not correct
                            )
                        )
                        s = tf.nn.softmax(a)

                h_i =
                W_h =
                a_i_t = tf.matmul()
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
