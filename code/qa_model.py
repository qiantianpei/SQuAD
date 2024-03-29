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

"""This file defines the top-level model"""

from __future__ import absolute_import
from __future__ import division 

import time
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import embedding_ops

from evaluate import exact_match_score, f1_score
from data_batcher import get_batch_generator
from pretty_print import print_example
from modules import RNNEncoder, SimpleSoftmaxLayer, BasicAttn, MultiAttn, SelfAttn,PntNet, masked_softmax

logging.basicConfig(level=logging.INFO)


class QAModel(object):
    """Top-level Question Answering module"""

    def __init__(self, FLAGS, id2word, word2id, emb_matrix, char2id, id2char):
        """
        Initializes the QA model.

        Inputs:
          FLAGS: the flags passed in from main.py
          id2word: dictionary mapping word idx (int) to word (string)
          word2id: dictionary mapping word (string) to word idx (int)
          emb_matrix: numpy array shape (400002, embedding_size) containing pre-traing GloVe embeddings
        """
        print "Initializing the QAModel..."
        self.FLAGS = FLAGS
        self.id2word = id2word
        self.word2id = word2id
        self.char2id = char2id
        self.id2char = id2char

        # Add all parts of the graph
        with tf.variable_scope("QAModel", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            print "add placeholder"
            self.add_placeholders()
            print "add embbding"
            self.add_embedding_layer(emb_matrix)
            print "Build graph"
            self.build_graph()
            print "add loss"
            self.add_loss()

        # Define trainable parameters, gradient, gradient norm, and clip by gradient norm
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        self.gradient_norm = tf.global_norm(gradients)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        self.param_norm = tf.global_norm(params)

        # Define optimizer and updates
        # (updates is what you need to fetch in session.run to do a gradient update)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        opt = tf.train.AdadeltaOptimizer(0.5, rho=0.999)
        #opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # you can try other optimizers
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        
        print "done loss"
        #self.updates = tf.train.AdadeltaOptimizer(1.0, rho=0.95, epsilon=1e-06).minimize(self.loss)
        # Define savers (for checkpointing) and summaries (for tensorboard)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.keep)
        self.bestmodel_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.summaries = tf.summary.merge_all()
        print "done init"


    def add_placeholders(self):
        """
        Add placeholders to the graph. Placeholders are used to feed in inputs.
        """
        # Add placeholders for inputs.
        # These are all batch-first: the None corresponds to batch_size and
        # allows you to run the same model with variable batch_size
        self.context_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.context_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len])
        self.qn_ids = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.qn_mask = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len])
        self.ans_span = tf.placeholder(tf.int32, shape=[None, 2])

        self.context_ids_c = tf.placeholder(tf.int32, shape=[None, self.FLAGS.context_len, self.FLAGS.word_len])
        self.qn_ids_c = tf.placeholder(tf.int32, shape=[None, self.FLAGS.question_len, self.FLAGS.word_len])

        # Add a placeholder to feed in the keep probability (for dropout).
        # This is necessary so that we can instruct the model to use dropout when training, but not when testing
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())


    def add_embedding_layer(self, emb_matrix):
        """
        Adds word embedding layer to the graph.

        Inputs:
          emb_matrix: shape (400002, embedding_size).
            The GloVe vectors, plus vectors for PAD and UNK.
        """
        with vs.variable_scope("embeddings"):

            # Note: the embedding matrix is a tf.constant which means it's not a trainable parameter
            embedding_matrix = tf.constant(emb_matrix, dtype=tf.float32, name="emb_matrix") # shape (400002, embedding_size)
            
            ##### my change
            embedding_matrix_c = tf.get_variable("emb_matrix_c", dtype=tf.float32, 
                shape = [len(self.char2id), self.FLAGS.embedding_size_c], initializer=tf.contrib.layers.xavier_initializer())
            ##### 

            # Get the word embeddings for the context and question,
            # using the placeholders self.context_ids and self.qn_ids
            self.context_embs = embedding_ops.embedding_lookup(embedding_matrix, self.context_ids) # shape (batch_size, context_len, embedding_size)
            self.qn_embs = embedding_ops.embedding_lookup(embedding_matrix, self.qn_ids) # shape (batch_size, question_len, embedding_size)

            ##### my change
            self.context_embs_c_raw = embedding_ops.embedding_lookup(embedding_matrix_c, self.context_ids_c) # shape (batch_size, context_len, word_len, embedding_size_c)
            self.qn_embs_c_raw = embedding_ops.embedding_lookup(embedding_matrix_c, self.qn_ids_c) # shape (batch_size, question_len, word_len, embedding_size_c)
            ##### 


    def build_graph(self):
        """Builds the main part of the graph for the model, starting from the input embeddings to the final distributions for the answer span.

        Defines:
          self.logits_start, self.logits_end: Both tensors shape (batch_size, context_len).
            These are the logits (i.e. values that are fed into the softmax function) for the start and end distribution.
            Important: these are -large in the pad locations. Necessary for when we feed into the cross entropy function.
          self.probdist_start, self.probdist_end: Both shape (batch_size, context_len). Each row sums to 1.
            These are the result of taking (masked) softmax of logits_start and logits_end.
        """

        # Use a RNN to get hidden states for the context and the question
        # Note: here the RNNEncoder is shared (i.e. the weights are the same)
        # between the context and the question.


        ##### my change: character-level CNN
        with vs.variable_scope("CharCNN"):
            context_embs_c_raw = tf.reshape(self.context_embs_c_raw, [-1, self.FLAGS.word_len, self.FLAGS.embedding_size_c]) # shape (batch_size * context_len, word_len, embedding_size)
            context_embs_c_raw = tf.nn.dropout(context_embs_c_raw, self.keep_prob)
            
            context_embs_c_conv = tf.layers.conv1d(inputs = context_embs_c_raw, filters = self.FLAGS.filters, kernel_size = self.FLAGS.kernel_size, padding = 'same', name = 'char_conv', reuse = None) # shape (batch_size * context_len, word_len, filters)
            #assert context_embs_c_conv.shape == [self.FLAGS.batch_size * self.FLAGS.context_len, self.FLAGS.word_len, self.FLAGS.filters]
            
            context_embs_c_pool = tf.layers.max_pooling1d(inputs = context_embs_c_conv, pool_size = self.FLAGS.word_len, strides = self.FLAGS.word_len) # shape (batch_size * context_len, 1, filters)
            #assert context_embs_c_pool.shape == [self.FLAGS.batch_size * self.FLAGS.context_len, 1, self.FLAGS.filters]
            
            context_embs_c = tf.reshape(context_embs_c_pool, [-1, self.FLAGS.context_len, self.FLAGS.filters]) # shape (batch_size , context_len, filters)
            #assert context_embs_c.shape == [self.FLAGS.batch_size, self.FLAGS.context_len, self.FLAGS.filters]

            #tf.get_variable_scope().reuse_variables()
            qn_embs_c_raw = tf.reshape(self.qn_embs_c_raw, [-1, self.FLAGS.word_len, self.FLAGS.embedding_size_c]) # shape (batch_size * question_len, word_len, embedding_size)
            qn_embs_c_raw = tf.nn.dropout(qn_embs_c_raw, self.keep_prob)
            
            qn_embs_c_conv = tf.layers.conv1d(inputs = qn_embs_c_raw, filters = self.FLAGS.filters, kernel_size = self.FLAGS.kernel_size, padding = 'same', name = 'char_conv', reuse = True) # shape (batch_size * question_len, word_len, filters)
            #assert qn_embs_c_conv.shape == [self.FLAGS.batch_size * self.FLAGS.question_len, self.FLAGS.word_len, self.FLAGS.filters]
            
            qn_embs_c_pool = tf.layers.max_pooling1d(inputs = qn_embs_c_conv, pool_size = self.FLAGS.word_len, strides = self.FLAGS.word_len) # shape (batch_size * question_len, 1, filters)
            #assert qn_embs_c_pool.shape == [self.FLAGS.batch_size * self.FLAGS.question_len, 1, self.FLAGS.filters]

            qn_embs_c = tf.reshape(qn_embs_c_pool, [-1, self.FLAGS.question_len, self.FLAGS.filters]) # shape (batch_size , question_len, filters)
            #assert qn_embs_c.shape == [self.FLAGS.batch_size, self.FLAGS.question_len, self.FLAGS.filters]

            context_embs_concat = tf.concat([self.context_embs, context_embs_c], axis = 2) # shape (batch_size , context_len, embedding_size + filters)
            qn_embs_concat = tf.concat([self.qn_embs, qn_embs_c], axis = 2) # shape (batch_size , question_len, embedding_size + filters)
        #####

        with vs.variable_scope("Contextual"):
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            context_hiddens = encoder.build_graph(context_embs_concat, self.context_mask) # (batch_size, context_len, 2 * hidden_size)
            question_hiddens = encoder.build_graph(qn_embs_concat, self.qn_mask)

            assert context_hiddens.shape[1:] == [self.FLAGS.context_len, 2 * self.FLAGS.hidden_size]
            assert question_hiddens.shape[1:] == [self.FLAGS.question_len, 2 * self.FLAGS.hidden_size]


        with vs.variable_scope("attention-flow"):
            w1 = tf.get_variable("w1", shape = [2 * self.FLAGS.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            w2 = tf.get_variable("w2", shape = [2 * self.FLAGS.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            w3 = tf.get_variable("w3", shape = [2 * self.FLAGS.hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())

            w1c = self.mat_weight_mul(context_hiddens, w1) # ((batch_size, context_len, 1)
            assert w1c.shape[1:] == [self.FLAGS.context_len, 1]

            w2q = self.mat_weight_mul(question_hiddens, w2) # ((batch_size, question_len, 1)
            assert w2q.shape[1:] == [self.FLAGS.question_len, 1]

            cq = tf.expand_dims(context_hiddens, 2) * tf.expand_dims(question_hiddens, 1) # ((batch_size, context_len, question_len, 2 * hidden_size)
            assert cq.shape[1:] == [self.FLAGS.context_len, self.FLAGS.question_len, 2 * self.FLAGS.hidden_size]

            w3cq = self.mat_weight_mul(tf.reshape(cq, [-1, self.FLAGS.context_len, 2 * self.FLAGS.hidden_size]), w2) # ((batch_size * context_len, question_len, 1)
            w3cq = tf.reshape(w3cq, [-1, self.FLAGS.context_len, self.FLAGS.question_len]) # ((batch_size, context_len, question_len)
            assert w3cq.shape[1:] == [self.FLAGS.context_len, self.FLAGS.question_len]


            S = w1c + tf.transpose(w2q, perm = [0, 2, 1]) + w3cq # (batch_size, context_len, question_len)
            assert S.shape[1:] == [self.FLAGS.context_len, self.FLAGS.question_len]

            attn_logits_mask = tf.expand_dims(self.qn_mask, 1) # shape (batch_size, 1, question_len)
            _, attn_C2Q = masked_softmax(S, attn_logits_mask, 2) # shape (batch_size, context_len, question_len)
            self.attn_C2Q = attn_C2Q

            assert attn_C2Q.shape[1:] == [self.FLAGS.context_len, self.FLAGS.question_len]

            # Use attention distribution to take weighted sum of values
            a = tf.matmul(attn_C2Q, question_hiddens) # shape (batch_size, context_len, 2 * hidden_size)
            assert a.shape[1:] == [self.FLAGS.context_len, 2 * self.FLAGS.hidden_size]

            # Apply dropout
            a = tf.nn.dropout(a, self.keep_prob)

            m = tf.reduce_max(S, axis = 2) # (batch_size, context_len)
            #m = tf.squeeze(m, [2]) # (batch_size, context_len)
            assert m.shape[1:] == [self.FLAGS.context_len]

            _, attn_Q2C = masked_softmax(m, self.context_mask, 1) # (batch_size, context_len)
            attn_Q2C = tf.expand_dims(attn_Q2C, 1) # (batch_size, 1, context_len)
            assert attn_Q2C.shape[1:] == [1, self.FLAGS.context_len]
            self.attn_Q2C = attn_Q2C

            c_prime = tf.matmul(attn_Q2C, context_hiddens) # (batch_size, 1, 2 * hidden_size)
            c_prime = tf.nn.dropout(c_prime, self.keep_prob)
            #c = tf.tile(c, [1, self.FLAGS.context_len, 1]) # (batch_size, context_len, 2 * hidden_size)

            # self_attn
            self_attn = MultiAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            _, d = self_attn.build_graph(context_hiddens, self.context_mask, context_hiddens)

            b = tf.concat([context_hiddens, a, context_hiddens * a, context_hiddens * c_prime, d], 2) # (batch_size, context_len, 10 * hidden_size)
            assert b.shape[1:] == [self.FLAGS.context_len, 10 * self.FLAGS.hidden_size]

        with vs.variable_scope("modelling1"):
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            M = encoder.build_graph(b, self.context_mask)

        with vs.variable_scope("modelling2"):
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            M = encoder.build_graph(M, self.context_mask)


        # with vs.variable_scope("selfAttn"):
            # W_vP1 = tf.get_variable('W_vP1', shape = [2 * self.FLAGS.hidden_size, self.FLAGS.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            # W_vP2 = tf.get_variable('W_VP2', shape = [2 * self.FLAGS.hidden_size, self.FLAGS.hidden_size], initializer = tf.contrib.layers.xavier_initializer())
            # v = tf.get_variable('v', shape = [self.FLAGS.hidden_size, 1],  initializer = tf.contrib.layers.xavier_initializer())

            # W_vP1_v_P = tf.expand_dims(self.mat_weight_mul(M, W_vP1), 1) # (batch_size, 1, context_len, hidden_size)
            # W_vP2_v_P = tf.expand_dims(self.mat_weight_mul(M, W_vP1), 2) # (batch_size, context_len, 1, hidden_size)

            # tanh = tf.tanh(W_vP1_v_P + W_vP2_v_P) # (batch_size, context_len, context_len, hidden_size)
            # assert tanh.shape[1:] == [self.FLAGS.context_len, self.FLAGS.context_len, self.FLAGS.hidden_size]

            # s = self.mat_weight_mul(tf.reshape(tanh, [-1, self.FLAGS.context_len, self.FLAGS.hidden_size]), v) # (batch_size * context_len, context_len, 1)
            # s = tf.reshape(s, [-1, self.FLAGS.context_len, self.FLAGS.context_len]) # (batch_size, context_len, context_len)
            # assert s.shape[1:] == [self.FLAGS.context_len, self.FLAGS.context_len]

            # self_mask = tf.expand_dims(self.context_mask, 1) # shape (batch_size, 1, context_len)
            # _, a = masked_softmax(s, self_mask, 2) # shape (batch_size, context_len, context_len)
            # assert a.shape[1:] == [self.FLAGS.context_len, self.FLAGS.context_len]

            # c = tf.matmul(a, M)
            # c = tf.nn.dropout(c, self.keep_prob)
            # Mc = tf.concat([M, c], 2) # (batch_size, context_len, hidden_size * 4)
            # assert Mc.shape[1:] == [self.FLAGS.context_len, 4 * self.FLAGS.hidden_size]

            # encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)

            # M = encoder.build_graph(Mc, self.context_mask) # (batch_size, context_len, hidden_size * 2)
            # assert M.shape[1:] == [self.FLAGS.context_len, 2 * self.FLAGS.hidden_size]

            # attn_layer = BasicAttn(self.keep_prob, self.FLAGS.hidden_size*2, self.FLAGS.hidden_size*2)
            # _, M = attn_layer.build_graph(M, self.context_mask, M)

        with vs.variable_scope("StartDist"):
            GM = tf.concat([b, M], 2)
            assert GM.shape[1:] == [self.FLAGS.context_len, 12 * self.FLAGS.hidden_size]

            softmax_layer_start = SimpleSoftmaxLayer(self.keep_prob)
            self.logits_start, self.probdist_start = softmax_layer_start.build_graph(GM, self.context_mask)

        # Use softmax layer to compute probability distribution for end location
        # Note this produces self.logits_end and self.probdist_end, both of which have shape (batch_size, context_len)
        with vs.variable_scope("EndDist"):
            encoder = RNNEncoder(self.FLAGS.hidden_size, self.keep_prob)
            M2 = encoder.build_graph(M, self.context_mask)
            GM2 = tf.concat([b, M2], 2)
            assert GM2.shape[1:] == [self.FLAGS.context_len, 12 * self.FLAGS.hidden_size]

            softmax_layer_end = SimpleSoftmaxLayer(self.keep_prob)
            self.logits_end, self.probdist_end = softmax_layer_end.build_graph(GM2, self.context_mask)
        

    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert(mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])

    def add_loss(self):
        """
        Add loss computation to the graph.

        Uses:
          self.logits_start: shape (batch_size, context_len)
            IMPORTANT: Assumes that self.logits_start is masked (i.e. has -large in masked locations).
            That's because the tf.nn.sparse_softmax_cross_entropy_with_logits
            function applies softmax and then computes cross-entropy loss.
            So you need to apply masking to the logits (by subtracting large
            number in the padding location) BEFORE you pass to the
            sparse_softmax_cross_entropy_with_logits function.

          self.ans_span: shape (batch_size, 2)
            Contains the gold start and end locations

        Defines:
          self.loss_start, self.loss_end, self.loss: all scalar tensors
        """
        with vs.variable_scope("loss"):

            # Calculate loss for prediction of start position
            loss_start = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_start, labels=self.ans_span[:, 0]) # loss_start has shape (batch_size)
            self.loss_start = tf.reduce_mean(loss_start) # scalar. avg across batch
            tf.summary.scalar('loss_start', self.loss_start) # log to tensorboard

            # Calculate loss for prediction of end position
            loss_end = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits_end, labels=self.ans_span[:, 1])
            self.loss_end = tf.reduce_mean(loss_end)
            tf.summary.scalar('loss_end', self.loss_end)

            # Add the two losses
            self.loss = self.loss_start + self.loss_end
            tf.summary.scalar('loss', self.loss)


    def run_train_iter(self, session, batch, summary_writer):
        """
        This performs a single training iteration (forward pass, loss computation, backprop, parameter update)

        Inputs:
          session: TensorFlow session
          batch: a Batch object
          summary_writer: for Tensorboard

        Returns:
          loss: The loss (averaged across the batch) for this batch.
          global_step: The current number of training iterations we've done
          param_norm: Global norm of the parameters
          gradient_norm: Global norm of the gradients
        """
        # Match up our input data with the placeholders
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        input_feed[self.keep_prob] = 1.0 - self.FLAGS.dropout # apply dropout

        ##### my change
        input_feed[self.context_ids_c] = batch.context_ids_c
        input_feed[self.qn_ids_c] = batch.qn_ids_c
        #####

        # output_feed contains the things we want to fetch.
        output_feed = [self.updates, self.summaries, self.loss, self.global_step, self.param_norm, self.gradient_norm]

        # Run the model
        [_, summaries, loss, global_step, param_norm, gradient_norm] = session.run(output_feed, input_feed)
        #x = session.run([self.context_hiddens, self.question_hiddens, self.a_t, self.a_t2, self.logits_start, self.probdist_start, self.logits_end, self.probdist_end, self.a], input_feed)

        #print x

        # All summaries in the graph are added to Tensorboard
        summary_writer.add_summary(summaries, global_step)

        return loss, global_step, param_norm, gradient_norm


    def get_loss(self, session, batch):
        """
        Run forward-pass only; get loss.

        Inputs:
          session: TensorFlow session
          batch: a Batch object

        Returns:
          loss: The loss (averaged across the batch) for this batch
        """

        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        input_feed[self.ans_span] = batch.ans_span
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        ##### my change
        input_feed[self.context_ids_c] = batch.context_ids_c
        input_feed[self.qn_ids_c] = batch.qn_ids_c
        #####

        output_feed = [self.loss]

        [loss] = session.run(output_feed, input_feed)

        return loss


    def get_prob_dists(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        ##### my change
        input_feed[self.context_ids_c] = batch.context_ids_c
        input_feed[self.qn_ids_c] = batch.qn_ids_c
        #####

        output_feed = [self.probdist_start, self.probdist_end]
        #print session.run(output_feed, input_feed)
        [probdist_start, probdist_end] = session.run(output_feed, input_feed)
        return probdist_start, probdist_end

    def get_attn(self, session, batch):
        """
        Run forward-pass only; get probability distributions for start and end positions.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          probdist_start and probdist_end: both shape (batch_size, context_len)
        """
        input_feed = {}
        input_feed[self.context_ids] = batch.context_ids
        input_feed[self.context_mask] = batch.context_mask
        input_feed[self.qn_ids] = batch.qn_ids
        input_feed[self.qn_mask] = batch.qn_mask
        # note you don't supply keep_prob here, so it will default to 1 i.e. no dropout

        ##### my change
        input_feed[self.context_ids_c] = batch.context_ids_c
        input_feed[self.qn_ids_c] = batch.qn_ids_c
        #####

        output_feed = [self.attn_C2Q, self.attn_Q2C]
        #print session.run(output_feed, input_feed)
        [attn_C2Q, attn_Q2C] = session.run(output_feed, input_feed)
        return attn_C2Q, attn_Q2C


    def get_start_end_pos(self, session, batch):
        """
        Run forward-pass only; get the most likely answer span.

        Inputs:
          session: TensorFlow session
          batch: Batch object

        Returns:
          start_pos, end_pos: both numpy arrays shape (batch_size).
            The most likely start and end positions for each example in the batch.
        """
        # Get start_dist and end_dist, both shape (batch_size, context_len)
        start_dist, end_dist = self.get_prob_dists(session, batch)

        # Take argmax to get start_pos and end_post, both shape (batch_size)
        # start_pos = np.argmax(start_dist, axis=1)
        # end_pos = np.argmax(end_dist, axis=1)

        prob = []
        span = 30
        for i in range(self.FLAGS.context_len - span):
            for j in range(span):
                prob.append(start_dist[:, i] * end_dist[:, i+j])
        prob = np.stack(prob, axis = 1)
        argmax_idx = np.argmax(prob, axis=1)
        start_pos = argmax_idx // span
        end_pos = start_pos + np.mod(argmax_idx, span)

        return start_pos, end_pos


    def get_dev_loss(self, session, dev_context_path, dev_qn_path, dev_ans_path):
        """
        Get loss for entire dev set.

        Inputs:
          session: TensorFlow session
          dev_qn_path, dev_context_path, dev_ans_path: paths to the dev.{context/question/answer} data files

        Outputs:
          dev_loss: float. Average loss across the dev set.
        """
        logging.info("Calculating dev loss...")
        tic = time.time()
        loss_per_batch, batch_lengths = [], []

        # Iterate over dev set batches
        # Note: here we set discard_long=True, meaning we discard any examples
        # which are longer than our context_len or question_len.
        # We need to do this because if, for example, the true answer is cut
        # off the context, then the loss function is undefined.
        for batch in get_batch_generator(self.word2id, self.char2id, dev_context_path, dev_qn_path, dev_ans_path, self.FLAGS.batch_size, 
            context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, word_len = self.FLAGS.word_len, discard_long=True):

            # Get loss for this batch
            loss = self.get_loss(session, batch)
            curr_batch_size = batch.batch_size
            loss_per_batch.append(loss * curr_batch_size)
            batch_lengths.append(curr_batch_size)

        # Calculate average loss
        total_num_examples = sum(batch_lengths)
        toc = time.time()
        print "Computed dev loss over %i examples in %.2f seconds" % (total_num_examples, toc-tic)

        # Overall loss is total loss divided by total number of examples
        dev_loss = sum(loss_per_batch) / float(total_num_examples)

        return dev_loss


    def check_f1_em(self, session, context_path, qn_path, ans_path, dataset, num_samples=100, print_to_screen=False):
        """
        Sample from the provided (train/dev) set.
        For each sample, calculate F1 and EM score.
        Return average F1 and EM score for all samples.
        Optionally pretty-print examples.

        Note: This function is not quite the same as the F1/EM numbers you get from "official_eval" mode.
        This function uses the pre-processed version of the e.g. dev set for speed,
        whereas "official_eval" mode uses the original JSON. Therefore:
          1. official_eval takes your max F1/EM score w.r.t. the three reference answers,
            whereas this function compares to just the first answer (which is what's saved in the preprocessed data)
          2. Our preprocessed version of the dev set is missing some examples
            due to tokenization issues (see squad_preprocess.py).
            "official_eval" includes all examples.

        Inputs:
          session: TensorFlow session
          qn_path, context_path, ans_path: paths to {dev/train}.{question/context/answer} data files.
          dataset: string. Either "train" or "dev". Just for logging purposes.
          num_samples: int. How many samples to use. If num_samples=0 then do whole dataset.
          print_to_screen: if True, pretty-prints each example to screen

        Returns:
          F1 and EM: Scalars. The average across the sampled examples.
        """
        logging.info("Calculating F1/EM for %s examples in %s set..." % (str(num_samples) if num_samples != 0 else "all", dataset))

        f1_total = 0.
        em_total = 0.
        example_num = 0

        tic = time.time()

        # Note here we select discard_long=False because we want to sample from the entire dataset
        # That means we're truncating, rather than discarding, examples with too-long context or questions
        for batch in get_batch_generator(self.word2id, self.char2id, context_path, qn_path, ans_path, self.FLAGS.batch_size, 
            context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, word_len=self.FLAGS.word_len, discard_long=False):

            pred_start_pos, pred_end_pos = self.get_start_end_pos(session, batch)

            # Convert the start and end positions to lists length batch_size
            pred_start_pos = pred_start_pos.tolist() # list length batch_size
            pred_end_pos = pred_end_pos.tolist() # list length batch_size


            # probdist_start, probdist_end = self.get_prob_dists(session, batch)
            # attn_C2Q ,attn_Q2C = self.get_attn(session, batch)
            # attn_C2Q = attn_C2Q.tolist()
            # attn_Q2C = attn_Q2C.tolist()
            # probdist_start = probdist_start.tolist()
            # probdist_end = probdist_end.tolist()

            #for ex_idx, (pred_ans_start, pred_ans_end, prob_ans_start, prob_ans_end, C2Q, Q2C, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, probdist_start, probdist_end, attn_C2Q, attn_Q2C, batch.ans_tokens)):
            for ex_idx, (pred_ans_start, pred_ans_end, true_ans_tokens) in enumerate(zip(pred_start_pos, pred_end_pos, batch.ans_tokens)):
                example_num += 1

                # Get the predicted answer
                # Important: batch.context_tokens contains the original words (no UNKs)
                # You need to use the original no-UNK version when measuring F1/EM
                pred_ans_tokens = batch.context_tokens[ex_idx][pred_ans_start : pred_ans_end + 1]
                pred_answer = " ".join(pred_ans_tokens)

                # Get true answer (no UNKs)
                true_answer = " ".join(true_ans_tokens)

                # Calc F1/EM
                f1 = f1_score(pred_answer, true_answer)
                em = exact_match_score(pred_answer, true_answer)
                f1_total += f1
                em_total += em

                # Optionally pretty-print
                if print_to_screen:
                    print_example(self.word2id, batch.context_tokens[ex_idx], batch.qn_tokens[ex_idx], batch.ans_span[ex_idx, 0], batch.ans_span[ex_idx, 1], pred_ans_start, pred_ans_end, true_answer, pred_answer, f1, em)
                    # print "C2Q"
                    # print C2Q
                    # print 
                    # print 
                    # print "Q2C"
                    # print Q2C
                    # print 
                    # print
                    # print "prob_start"
                    # print prob_ans_start 
                    # print 
                    # print
                    # print "prob_end"
                    # print prob_ans_end

                if num_samples != 0 and example_num >= num_samples:
                    break

            if num_samples != 0 and example_num >= num_samples:
                break

        f1_total /= example_num
        em_total /= example_num

        toc = time.time()
        logging.info("Calculating F1/EM for %i examples in %s set took %.2f seconds" % (example_num, dataset, toc-tic))

        return f1_total, em_total


    def train(self, session, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path):
        """
        Main training loop.

        Inputs:
          session: TensorFlow session
          {train/dev}_{qn/context/ans}_path: paths to {train/dev}.{context/question/answer} data files
        """

        # Print number of model parameters
        print "train start"
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retrieval took %f secs)" % (num_params, toc - tic))

        # We will keep track of exponentially-smoothed loss
        exp_loss = None

        # Checkpoint management.
        # We keep one latest checkpoint, and one best checkpoint (early stopping)
        checkpoint_path = os.path.join(self.FLAGS.train_dir, "qa.ckpt")
        bestmodel_dir = os.path.join(self.FLAGS.train_dir, "best_checkpoint")
        bestmodel_ckpt_path = os.path.join(bestmodel_dir, "qa_best.ckpt")
        best_dev_f1 = None
        best_dev_em = None

        # for TensorBoard
        summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir, session.graph)

        epoch = 0

        logging.info("Beginning training loop...")
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            epoch += 1
            epoch_tic = time.time()

            # Loop over batches
            for batch in get_batch_generator(self.word2id, self.char2id, train_context_path, train_qn_path, train_ans_path, self.FLAGS.batch_size, 
                context_len=self.FLAGS.context_len, question_len=self.FLAGS.question_len, word_len = self.FLAGS.word_len, discard_long=True):

                # Run training iteration
                iter_tic = time.time()
                loss, global_step, param_norm, grad_norm = self.run_train_iter(session, batch, summary_writer)
                iter_toc = time.time()
                iter_time = iter_toc - iter_tic

                # Update exponentially-smoothed loss
                if not exp_loss: # first iter
                    exp_loss = loss
                else:
                    exp_loss = 0.99 * exp_loss + 0.01 * loss

                # Sometimes print info to screen
                if global_step % self.FLAGS.print_every == 0:
                    logging.info(
                        'epoch %d, iter %d, loss %.5f, smoothed loss %.5f, grad norm %.5f, param norm %.5f, batch time %.3f' %
                        (epoch, global_step, loss, exp_loss, grad_norm, param_norm, iter_time))

                # Sometimes save model
                if global_step % self.FLAGS.save_every == 0:
                    logging.info("Saving to %s..." % checkpoint_path)
                    self.saver.save(session, checkpoint_path, global_step=global_step)

                # Sometimes evaluate model on dev loss, train F1/EM and dev F1/EM
                if global_step % self.FLAGS.eval_every == 0:

                    # Get loss for entire dev set and log to tensorboard
                    dev_loss = self.get_dev_loss(session, dev_context_path, dev_qn_path, dev_ans_path)
                    logging.info("Epoch %d, Iter %d, dev loss: %f" % (epoch, global_step, dev_loss))
                    write_summary(dev_loss, "dev/loss", summary_writer, global_step)


                    # Get F1/EM on train set and log to tensorboard
                    train_f1, train_em = self.check_f1_em(session, train_context_path, train_qn_path, train_ans_path, "train", num_samples=1000)
                    logging.info("Epoch %d, Iter %d, Train F1 score: %f, Train EM score: %f" % (epoch, global_step, train_f1, train_em))
                    write_summary(train_f1, "train/F1", summary_writer, global_step)
                    write_summary(train_em, "train/EM", summary_writer, global_step)


                    # Get F1/EM on dev set and log to tensorboard
                    dev_f1, dev_em = self.check_f1_em(session, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=0)
                    logging.info("Epoch %d, Iter %d, Dev F1 score: %f, Dev EM score: %f" % (epoch, global_step, dev_f1, dev_em))
                    write_summary(dev_f1, "dev/F1", summary_writer, global_step)
                    write_summary(dev_em, "dev/EM", summary_writer, global_step)


                    # Early stopping based on dev EM. You could switch this to use F1 instead.
                    if best_dev_em is None or dev_em > best_dev_em:
                        best_dev_em = dev_em
                        logging.info("Saving to %s..." % bestmodel_ckpt_path)
                        self.bestmodel_saver.save(session, bestmodel_ckpt_path, global_step=global_step)


            epoch_toc = time.time()
            logging.info("End of epoch %i. Time for epoch: %f" % (epoch, epoch_toc-epoch_tic))

        sys.stdout.flush()



def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)
