# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output

class MultiAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("MultiAttn", reuse = None):

            # Calculate attention distribution
            W = tf.get_variable('W', shape = [self.key_vec_size, self.value_vec_size], initializer=tf.contrib.layers.xavier_initializer()) # (key_vec_size, value_vec_size)
            W = tf.expand_dims(W, 0) # (1, key_vec_size, value_vec_size)
            W = tf.tile(W, [tf.shape(values)[0], 1, 1]) # (batch_size, key_vec_size, value_vec_size)

            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(tf.matmul(keys, W), values_t) / tf.sqrt(self.key_vec_size * 1.) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


class GatedAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size
        self.hidden_size = hidden_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("GatedAttn"):

            v_P = []
            W_uQ = tf.get_variable('W_uQ', shape = [self.value_vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_uP = tf.get_variable('W_uP', shape = [self.key_vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_vP = tf.get_variable('W_vP', shape = [self.key_vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            v = tf.get_variable('v', shape = [self.hidden_size, 1],  initializer = tf.contrib.layers.xavier_initializer())
            W_g = tf.get_variable('W_g', shape = [self.key_vec_size + self.value_vec_size, self.key_vec_size + self.value_vec_size], initializer = tf.contrib.layers.xavier_initializer())

            QP_match_cell = rnn_cell.GRUCell(self.hidden_size)
            QP_match_cell = DropoutWrapper(QP_match_cell, input_keep_prob=self.keep_prob)
            QP_match_state = QP_match_cell.zero_state(tf.shape(values)[0], tf.float32)

            for t in range(tf.shape(self.key)[1]): # context_len
                W_uQ_u_Q = tf.tensordot(values, W_uQ, 1) # (batch_size, q_len, hidden_size)
                W_uP_u_tP = tf.tensordot(keys[:,t:(t+1),:], W_uP, 1) # (batch_size, 1, hidden_size)

                if t == 0:
                    tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)
                else:
                    W_vP_v_t1P = tf.tensordot(tf.expand_dims(v_P[t-1], 1), W_uP, 1)
                    tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P)
                assert tanh.shape[1:] == [tf.shape(self.values)[1], self.hidden_siz]

                s_t = tf.squeeze(tf.tensordot(tanh, v, 1)) # (batch_size, q_len)
                _, a_t = masked_softmax(s_t, values_mask, 1) # (batch_size, q_len)
                c_t = tf.tensordot(tf.expand_dims(a_t, 1), values, 1)
                c_t = tf.nn.dropout(c_t, self.keep_prob)
                assert c_t.shape[1:] == [1, self.value_vec_size]
                u_tP_c_t = tf.concat([keys[:,t:(t+1),:], c_t], 2) # (batch_size, 1, self.value_vec_size + self.key_vec_size)
                assert u_tP_c_t.shape[1:] == [1, self.value_vec_size + self.key_vec_size]
                g_t = tf.tensordot(u_tP_c_t, W_g, 1) # (batch_size, 1, self.value_vec_size + self.key_vec_size)
                u_tP_c_t_star = tf.squeeze(u_tP_c_t * g_t) # (batch_size, self.value_vec_size + self.key_vec_size)

                with tf.variable_scope("QP_match"):
                    if t > 0:
                        tf.get_variable_scope().reuse_variables()
                    output, QPmatch_state = QP_match_cell(u_tP_c_t_star, QP_match_state)
                    v_P.append(output) # output: (batch_size, hidden_size)
            v_P = tf.stack(v_P, 1)
            assert v_P.shape[1:] = [tf.shape(self.key)[1], self.hidden_size]
            #v_P = tf.nn.dropout(v_P, self.keep_prob)
                    
        return v_P

class SelfAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, vec_size, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.vec_size = vec_size
        self.hidden_size = hidden_size

    def build_graph(self, values, values_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("SelfAttn"):

            W_vP1 = tf.get_variable('W_vP1', shape = [self.vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_vP2 = tf.get_variable('W_VP2', shape = [self.vec_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            v = tf.get_variable('v', shape = [self.hidden_size, 1],  initializer = tf.contrib.layers.xavier_initializer())

            W_vP1_v_P = tf.expand_dims(tf.tensordot(values, W_vP1, 1), 1) # (batch_size, 1, context_len, hidden_size)
            W_vP2_v_P = tf.expand_dims(tf.tensordot(values, W_vP2, 1), 2) # (batch_size, context_len, 1, hidden_size)

            tanh = tf.tanh(W_vP1_v_P + W_vP2_v_P) # (batch_size, context_len, context_len, hidden_size)
            s = tf.tensordot(tanh, v, 1) # (batch_size, context_len, context_len, 1)

            _, a = masked_softmax(s, tf.expand_dims(values_mask, 1), 2) # shape (batch_size, context_len, context_len)
            c = tf.matmul(a, values)
            c = tf.nn.dropout(c, self.keep_prob)
            v_P_c = tf.concat([values, c], 2) # (batch_size, context_len, context_size * 2)

            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)

            h_P = encoder.build_graph(v_P_c, values_mask) # (batch_size, context_len, hidden_size * 2)
                    
        return h_P


class PntNet(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, context_size, quesiton_size, hidden_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.question_size = question_size

    def build_graph(self, context, question, context_mask, question_mask):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("PntNet"):

            W_hP = tf.get_variable('W_hP', shape = [self.context_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_ha = tf.get_variable('W_ha', shape = [self.question_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_uQ = tf.get_variable('W_vP', shape = [self.question_size, self.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            W_vQ_V_rQ = tf.get_variable('W_vQ_V_rQ', shape = [1, 1, self.hidden_size],  initializer = tf.contrib.layers.xavier_initializer())
            v1 = tf.get_variable('v1', shape = [self.hidden_size, 1],  initializer = tf.contrib.layers.xavier_initializer())
            v2 = tf.get_variable('v2', shape = [self.hidden_size, 1],  initializer = tf.contrib.layers.xavier_initializer())

            ptr_cell = rnn_cell.GRUCell(self.hidden_size)
            ptr_cell = DropoutWrapper(ptr_cell, input_keep_prob=self.keep_prob)

            W_uQ_u_Q = tf.tensordot(values, W_uQ, 1) # (batch_size, q_len, hidden_size)
            tanh1 = tf.tanh(W_uQ_u_Q + W_vQ_V_rQ) # (batch_size, q_len, hidden_size)
            s1 = tf.squeeze(tf.tensordot(tanh1, v1, 1)) # (batch_size, q_len)
            _, a = masked_softmax(s1, question_mask, 1)
            r1 = tf.tensordot(tf.expand_dims(a_t, 1), question, 1)
            r1 = tf.nn.dropout(r1, self.keep_prob) # (batch_size, 1, question_size)

            W_ha_r1 = tf.tensordot(r1, W_ha, 1) # (batch_size, 1, hidden_size)
            W_hP_h_P1 = tf.tensordot(context, W_hP, 1) # (batch_size, c_len, hidden_size)

            tanh2 = tf.tanh(W_ha_r1 + W_hP_h_P1) # (batch_size, c_len, hidden_size)
            s2 = tf.squeeze(tf.tensordot(tanh2, v2, 1)) # (batch_size, c_len)
            logits_start, probdist_start = masked_softmax(s2, context_mask, 1)
            c = tf.squeeze(tf.tensordot(tf.expand_dims(probdist_start, 1), context, 1)) # (batch_size, context_size)
            r2, _ = ptr_cell(c, tf.squeeze(r1))

            W_ha_r2 = tf.tensordot(r2, W_ha, 1) # (batch_size, 1, hidden_size)
            W_hP_h_P2 = tf.tensordot(context, W_hP, 1) # (batch_size, c_len, hidden_size)

            tanh3 = tf.tanh(W_ha_r2 + W_hP_h_P2) # (batch_size, c_len, hidden_size)
            s3 = tf.squeeze(tf.tensordot(tanh3, v2, 1)) # (batch_size, c_len)
            logits_end, probdist_end = masked_softmax(s3, context_mask, 1) 
                    
        return logits_start, probdist_start, logits_end, probdist_end


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
