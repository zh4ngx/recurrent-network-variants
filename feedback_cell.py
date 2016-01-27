from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import array_ops


class FeedbackCell(rnn_cell.MultiRNNCell):
    """
    MultiRNNCell composed of stacked cells that interact across layers
    Based on http://arxiv.org/pdf/1502.02367v4.pdf
    """

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Conveniently the concatenation of all hidden states at t-1
            h_star_t_prev = state
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("Cell%d" % i):
                    cur_state = array_ops.slice(
                            state, [0, cur_state_pos], [-1, cell.state_size])
                    with tf.variable_scope("Global Reset"):
                        u_g = tf.get_variable("u_g", [self.state_size],
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1))
                        w_g = tf.get_variable("w_g", cell.state_size,
                                              initializer=tf.random_uniform_initializer(-0.1, 0.1))
                        g = tf.sigmoid(tf.mul(w_g, cur_inp) + tf.mul(u_g, h_star_t_prev))
                        cur_state = tf.reduce_sum(g * cur_state)

                    cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, cur_state)
                    new_states.append(new_state)

        return cur_inp, array_ops.concat(1, new_states)
