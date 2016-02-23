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
            x = inputs

            with tf.variable_scope("attention"):
                v = tf.get_variable("v", [self.output_size],
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                W_h = tf.get_variable("W_h", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                W_x = tf.get_variable("W_x", [self.input_size, self._num_units],
                                      initializer=tf.random_uniform_initializer(-0.1, 0.1))
                W_h_c = tf.get_variable("W_h_c", [self.input_size, self._num_units],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))

                # TODO: vectorize
                for i in range(0, current_timestep):
                    with tf.variable_scope(i):
                        a = tf.matmul(
                            v,
                            tf.tanh(
                                tf.matmul(W_h, h[:, :, i]) +
                                tf.matmul(W_x, x) +
                                tf.matmul(W_h_c, h[:, :, i])  # not correct
                            ),
                            name="a"
                        )
                a = tf.concat(1, [tf.get_variable(str(i) + "/a") for i in range(0, current_timestep)])

                # Attention Softmax
                s = tf.nn.softmax(a)

            with tf.variable_scope("memory"):
                # TODO: vectorize
                for i in range(0, current_timestep):
                    with tf.variable_scope(i):
                        m = s[i] * memory_tape[:, :, i]
                summation = tf.reduce_sum(tf.get_variable(str(i) + "/m") for i in range(0, current_timestep))

                h, c = tf.split(1, 2, summation)

            with tf.variable_scope("Input Gate"):
                W = tf.get_variable("W", [self.input_size, self._num_units],
                                    initializer=tf.random_uniform_initializer(-0.1, 0.1))

            # TODO: FINISH i, f, o, m GATES - based on x_t and memory-adapted h
            # TODO: FINISH final c output - based on gated c and memory-adapted c
            # TODO: FINISH final h output - based on gated c output

        return h, c
