import random
import os
import numpy as np

import nltk
import word2vec as w2v
from collections import Counter
from sklearn.decomposition import PCA
import json
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle

from keras.utils.theano_utils import shared_zeros, alloc_zeros_matrix

import utils
import nn_utils
from movieqa_importer import MovieQA

w2v_mqa_model_filename = 'models/movie_plots_1364.d-300.mc1.w2v'
#theano.config.floatX = 'float32'
floatX = theano.config.floatX


class DMN_untied:
    
    def __init__(self, stories, QAs, batch_size, story_v, learning_rate, word_vector_size, sent_vector_size, 
                dim, mode, answer_module, input_mask_mode, memory_hops, l2, story_source,
                normalize_attention, batch_norm, dropout, dropout_in, **kwargs):

        #print "==> not used params in DMN class:", kwargs.keys()
        self.learning_rate = learning_rate
        self.rng = np.random
        self.rng.seed(1234)
        mqa = MovieQA.DataLoader()
        ### Load Word2Vec model
        w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')
        self.w2v = w2v_model
        self.d_w2v = len(w2v_model.get_vector(w2v_model.vocab[1]))
        self.word_thresh = 1
        print "Loaded word2vec model: dim = %d | vocab-size = %d" % (self.d_w2v, len(w2v_model.vocab))
        ### Create vocabulary-to-index and index-to-vocabulary
        v2i = {'': 0, 'UNK':1}  # vocabulary to index
        QA_words, v2i = self.create_vocabulary(QAs, stories, v2i, w2v_vocab=w2v_model.vocab.tolist(), word_thresh=self.word_thresh)
        i2v = {v:k for k,v in v2i.iteritems()}
        self.vocab = v2i
        self.ivocab = i2v
        self.story_v = story_v
        self.word2vec = w2v_model
        self.word_vector_size = word_vector_size
        self.sent_vector_size = sent_vector_size
        self.dim = dim
        self.batch_size = batch_size
        self.mode = mode
        self.answer_module = answer_module
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.l2 = l2
        self.normalize_attention = normalize_attention
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_in = dropout_in

        #self.max_inp_sent_len = 0
        #self.max_q_len = 0

        ### Convert QAs and stories into numpy matrices (like in the bAbI data set)
        # storyM - Dictionary - indexed by imdb_key. Values are [num-sentence X max-num-words]
        # questionM - NP array - [num-question X max-num-words]
        # answerM - NP array - [num-question X num-answer-options X max-num-words]
        storyM, questionM, answerM = self.data_in_matrix_form(stories, QA_words, v2i)
        qinfo = self.associate_additional_QA_info(QAs)

        ### Split everything into train, val, and test data
        train_storyM = {k:v for k, v in storyM.iteritems() if k in mqa.data_split['train']}
        val_storyM   = {k:v for k, v in storyM.iteritems() if k in mqa.data_split['val']}
        test_storyM  = {k:v for k, v in storyM.iteritems() if k in mqa.data_split['test']}

        def split_train_test(long_list, QAs, trnkey='train', tstkey='val'):
            # Create train/val/test splits based on key
            train_split = [item for k, item in enumerate(long_list) if QAs[k].qid.startswith('train')]
            val_split = [item for k, item in enumerate(long_list) if QAs[k].qid.startswith('val')]
            test_split = [item for k, item in enumerate(long_list) if QAs[k].qid.startswith('test')]
            if type(long_list) == np.ndarray:
                return np.array(train_split), np.array(val_split), np.array(test_split)
            else:
                return train_split, val_split, test_split

        train_questionM, val_questionM, test_questionM = split_train_test(questionM, QAs)
        train_answerM,   val_answerM,   test_answerM,  = split_train_test(answerM, QAs)
        train_qinfo,     val_qinfo,     test_qinfo     = split_train_test(qinfo, QAs)

        QA_train = [qa for qa in QAs if qa.qid.startswith('train:')]
        QA_val   = [qa for qa in QAs if qa.qid.startswith('val:')]
        QA_test  = [qa for qa in QAs if qa.qid.startswith('test:')]

        #train_data = {'s':train_storyM, 'q':train_questionM, 'a':train_answerM, 'qinfo':train_qinfo}
        #val_data =   {'s':val_storyM,   'q':val_questionM,   'a':val_answerM,   'qinfo':val_qinfo}
        #test_data  = {'s':test_storyM,  'q':test_questionM,  'a':test_answerM,  'qinfo':test_qinfo}

        with open('train_split.json') as fid:
            trdev = json.load(fid)

        s_key = self.story_v.keys()
        self.train_range = [k for k, qi in enumerate(qinfo) if (qi['movie'] in trdev['train'] and qi['qid'] in s_key)]
        self.train_val_range   = [k for k, qi in enumerate(qinfo) if (qi['movie'] in trdev['dev'] and qi['qid'] in s_key)]
        self.val_range = [k for k, qi in enumerate(val_qinfo) if qi['qid'] in s_key]

        self.max_sent_len = max([sty.shape[0] for sty in self.story_v.values()])
        self.train_input = self.story_v
        self.train_val_input = self.story_v
        self.test_input = self.story_v
        self.train_q = train_questionM
        self.train_answer = train_answerM
        self.train_qinfo = train_qinfo
        self.train_val_q = train_questionM
        self.train_val_answer = train_answerM
        self.train_val_qinfo = train_qinfo
        self.test_q = val_questionM
        self.test_answer = val_answerM
        self.test_qinfo = val_qinfo
        
        """Setup some configuration parts of the model.
        """
        self.v2i = v2i
        self.vs = len(v2i)
        self.d_lproj = 300

        # define Look-Up-Table mask
        np_mask = np.vstack((np.zeros(self.d_w2v), np.ones((self.vs - 1, self.d_w2v))))
        T_mask = theano.shared(np_mask.astype(theano.config.floatX), name='LUT_mask')

        # setup Look-Up-Table to be Word2Vec
        self.pca_mat = None
        print "Initialize LUTs as word2vec and use linear projection layer"

        self.LUT = np.zeros((self.vs, self.d_w2v), dtype='float32')
        found_words = 0
        for w, v in self.v2i.iteritems():
            if w in self.w2v.vocab:  # all valid words are already in vocab or 'UNK'
                self.LUT[v] = self.w2v.get_vector(w)
                found_words += 1
            else:
                # LUT[v] = np.zeros((self.d_w2v))
                self.LUT[v] = self.rng.randn(self.d_w2v)
                self.LUT[v] = self.LUT[v] / (np.linalg.norm(self.LUT[v]) + 1e-6)

        print "Found %d / %d words" %(found_words, len(self.v2i))

        # word 0 is blanked out, word 1 is 'UNK'
        self.LUT[0] = np.zeros((self.d_w2v))

        # if linear projection layer is not the same shape as LUT, then initialize with PCA
        if self.d_lproj != self.LUT.shape[1]:
            pca = PCA(n_components=self.d_lproj, whiten=True)
            self.pca_mat = pca.fit_transform(self.LUT.T)  # 300 x 100?

        # setup LUT!
        self.T_w2v = theano.shared(self.LUT.astype(theano.config.floatX))

        self.train_input_mask = np_mask
        self.test_input_mask = np_mask
        #self.train_input, self.train_q, self.train_answer, self.train_input_mask = self._process_input(babi_train_raw)
        #self.test_input, self.test_q, self.test_answer, self.test_input_mask = self._process_input(babi_test_raw)
        self.vocab_size = len(self.vocab)

        self.input_var = T.tensor3('input_var')
        self.q_var = T.matrix('question_var')
        self.answer_var = T.tensor3('answer_var')
        self.input_mask_var = T.imatrix('input_mask_var')
        self.target = T.ivector('target')
        self.attentions = []

        #self.pe_matrix_in = self.pe_matrix(self.max_inp_sent_len)
        #self.pe_matrix_q = self.pe_matrix(self.max_q_len)

            
        print "==> building input module"

        #positional encoder weights
        self.W_pe = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))

        #biGRU input fusion weights
        self.W_inp_res_in_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_res_hid_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res_fwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_upd_hid_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd_fwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_hid_hid_fwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid_fwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_inp_res_in_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_res_hid_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res_bwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_upd_hid_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd_bwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.sent_vector_size))
        self.W_inp_hid_hid_bwd = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid_bwd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        #self.V_f = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        #self.V_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))

        self.inp_sent_reps = self.input_var

        #self.inp_sent_reps_stacked = T.stacklists(self.inp_sent_reps)
        #self.inp_c = self.input_module_full(self.inp_sent_reps_stacked)
        self.ans_reps = self.answer_var
        self.inp_c = self.input_module_full(self.inp_sent_reps)

        self.q_q = self.q_var
                
        print "==> creating parameters for memory module"
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        
        #self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        #self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, self.dim, 4 * self.dim + 0))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(self.memory_hops, 1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(self.memory_hops, 1,))


        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            self.mem_weight_num = int(iter - 1)
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in[self.mem_weight_num], self.W_mem_res_hid[self.mem_weight_num], self.b_mem_res[self.mem_weight_num], 
                                          self.W_mem_upd_in[self.mem_weight_num], self.W_mem_upd_hid[self.mem_weight_num], self.b_mem_upd[self.mem_weight_num],
                                          self.W_mem_hid_in[self.mem_weight_num], self.W_mem_hid_hid[self.mem_weight_num], self.b_mem_hid[self.mem_weight_num]))
        
        last_mem_raw = memory[-1]
        
        net = layers.InputLayer(shape=(self.batch_size, self.dim), input_var=last_mem_raw)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net)[0]
        
        print "==> building answer module"
        self.W_a = nn_utils.normal_param(std=0.1, shape=(300, self.dim))
        
        if self.answer_module == 'feedforward':
            self.temp = T.dot(self.ans_reps, self.W_a)
            self.prediction = nn_utils.softmax(T.dot(self.temp, last_mem))
        
        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
                
                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]
            
            # add conditional ending?
            dummy = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
            
            results, updates = theano.scan(fn=answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy)],
                n_steps=1)
            self.prediction = results[1][-1]
        
        else:
            raise Exception("invalid answer_module")
        
        
        print "==> collecting all parameters"
        self.params = [self.W_pe,
                  self.W_inp_res_in_fwd, self.W_inp_res_hid_fwd, self.b_inp_res_fwd, 
                  self.W_inp_upd_in_fwd, self.W_inp_upd_hid_fwd, self.b_inp_upd_fwd,
                  self.W_inp_hid_in_fwd, self.W_inp_hid_hid_fwd, self.b_inp_hid_fwd,
                  self.W_inp_res_in_bwd, self.W_inp_res_hid_bwd, self.b_inp_res_bwd, 
                  self.W_inp_upd_in_bwd, self.W_inp_upd_hid_bwd, self.b_inp_upd_bwd,
                  self.W_inp_hid_in_bwd, self.W_inp_hid_hid_bwd, self.b_inp_hid_bwd, 
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]

        if self.answer_module == 'recurrent':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
        
        
        print "==> building loss layer and computing updates"
        self.loss_ce = T.nnet.categorical_crossentropy(self.prediction, self.target)

        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = T.mean(self.loss_ce) + self.loss_l2
        
        #updates = lasagne.updates.adadelta(self.loss, self.params)
        #updates = lasagne.updates.adam(self.loss, self.params)
        updates = lasagne.updates.adam(self.loss, self.params, learning_rate=self.learning_rate, beta1=0.5) #from DCGAN paper
        #updates = lasagne.updates.adadelta(self.loss, self.params, learning_rate=0.0005)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.0003)
        
        self.attentions = T.stack(self.attentions)
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.target], 
                                            outputs=[self.prediction, self.loss, self.attentions],
                                            updates=updates,
                                            on_unused_input='warn',
                                            allow_input_downcast=True)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.target],
                                       outputs=[self.prediction, self.loss, self.attentions],
                                       on_unused_input='warn',
                                       allow_input_downcast=True)
    
    '''
    def pe_matrix(self, num_words):
        embedding_size = self.dim

        pe_matrix = np.ones((num_words, embedding_size))

        for j in range(num_words):
            for i in range(embedding_size):
                value = (i + 1. - (embedding_size + 1.) / 2.) * (j + 1. - (num_words + 1.) / 2.)
                pe_matrix[j,i] = float(value)
        pe_matrix = 1. + 4. * pe_matrix / (float(embedding_size) * num_words)

        return pe_matrix
    '''
    def pos_encodings(self, statement):
        statement = self.LUT[statement]
        num_words = len(statement)
        l = np.zeros(300)
        for j in range(num_words):
            for d in range(300):
                l[d] = (1-(1.+j)/num_words)-((d+1.)/300)*(1-2*(1.+j)/num_words)
            statement[j] = l[d] * statement[j]
        memories = sum(statement)
        return memories
    '''
    def sum_pos_encodings_q(self, statement):
        pe_matrix = self.pe_matrix_q
        pe_weights = pe_matrix * self.W_pe[statement]
        pe_weights = T.cast(pe_weights, floatX)
        memories = T.sum(pe_weights, axis=0)
        return memories
    
    def get_sentence_representation(self, statements):
        sent_rep, _ = theano.scan(fn = self.sum_pos_encodings,
            sequences = statements)
        return sent_rep
    '''
    def bi_GRU_fwd(self, x_fwd, prev_h):
        fwd_gru = self.GRU_update(prev_h, x_fwd, self.W_inp_res_in_fwd, self.W_inp_res_hid_fwd, self.b_inp_res_fwd, 
                                 self.W_inp_upd_in_fwd, self.W_inp_upd_hid_fwd, self.b_inp_upd_fwd,
                                 self.W_inp_hid_in_fwd, self.W_inp_hid_hid_fwd, self.b_inp_hid_fwd)
        '''
        if self.dropout_in > 0 and self.mode == 'train':
            fwd_gru_swap = fwd_gru.dimshuffle(('x', 0))
            net = layers.InputLayer(shape=(1, self.dim), input_var=fwd_gru_swap)
            net = layers.DropoutLayer(net, p=self.dropout_in)
            fwd_gru_d = layers.get_output(net)[0]
            fwd_gru = fwd_gru_d
        #'''
        return fwd_gru

    def bi_GRU_bwd(self, x_bwd, prev_h):
        bwd_gru = self.GRU_update(prev_h, x_bwd, self.W_inp_res_in_bwd, self.W_inp_res_hid_bwd, self.b_inp_res_bwd, 
                                 self.W_inp_upd_in_bwd, self.W_inp_upd_hid_bwd, self.b_inp_upd_bwd,
                                 self.W_inp_hid_in_bwd, self.W_inp_hid_hid_bwd, self.b_inp_hid_bwd)
        '''
        if self.dropout_in > 0 and self.mode == 'train':
            bwd_gru_swap = bwd_gru.dimshuffle(('x', 0))
            net = layers.InputLayer(shape=(1, self.dim), input_var=bwd_gru_swap)
            net = layers.DropoutLayer(net, p=self.dropout_in)
            bwd_gru_d = layers.get_output(net)[0]
            bwd_gru = bwd_gru_d
        #'''
        return bwd_gru

    def input_module_full(self, x):
        '''
        based on https://github.com/uyaseen/theano-recurrence/blob/master/model/gru.py
        based on Kyle_Kastner's comment: https://news.ycombinator.com/item?id=11237125
        '''
        x_fwd = x
        x_fwd = x_fwd.dimshuffle(1, 0, 2)               # sentences X batch-size X 300
        x_bwd = x_fwd
        x_bwd = x_bwd[::-1]
        tmp = theano.shared(np.zeros([self.batch_size, self.dim], dtype='float32'))

        h_fwd_gru, _ = theano.scan(fn=self.bi_GRU_fwd, 
                    #sequences=self.inp_sent_reps,
                    sequences=x_fwd,
                    outputs_info=T.zeros_like(tmp))
                    #outputs_info=T.zeros_like(self.W_inp_hid_hid))

        h_bwd_gru, _ = theano.scan(fn=self.bi_GRU_bwd, 
                    #sequences=self.inp_sent_reps,
                    sequences=x_bwd,
                    outputs_info=T.zeros_like(tmp))
                    #outputs_info=T.zeros_like(self.W_inp_hid_hid))

        h_bwd_gru = h_bwd_gru[::-1]

        '''
        #axis=0 and no transposes is original that works
        ctx = T.concatenate([h_fwd_gru, h_bwd_gru], axis=0)
        ht = ctx
        #'''

        '''
        #axis=1 and transpose parts & whole also works
        ctx = T.concatenate([h_fwd_gru.T, h_bwd_gru.T], axis=1)
        ht = ctx.T
        #'''

        '''
        #weighted sum version
        #h_t = T.dot(h_fwd_gru, self.V_f) + T.dot(h_bwd_gru, self.V_b)
        '''

        h_t = h_bwd_gru + h_fwd_gru

        return h_t 

    def episode_compute_z(self, fi, prev_g, mem, q_q):
        #euclid square version
        z = T.concatenate([fi * q_q, fi * mem, (fi - q_q) ** 2, (fi - mem) ** 2], axis=1)
        
        #T.abs_ version
        #z = T.concatenate([fi * q_q, fi * mem, T.abs_(fi - q_q), T.abs_(fi - mem)])

        l_1 = T.dot(z, self.W_1[self.mem_weight_num].T) + self.b_1[self.mem_weight_num]
        l_1 = T.tanh(l_1)
        l_2 = T.dot(l_1, self.W_2[self.mem_weight_num].T) + self.b_2[self.mem_weight_num]

        exp_l_2 = T.exp(l_2)

        return exp_l_2

    def episode_compute_g(self, z_i, z_all):
        G = z_i/(T.sum(z_all, axis=0))
        #G = G[0]
        return G

    def episode_attend(self, x, g, h):
        r = T.nnet.sigmoid(T.dot(x, self.W_mem_res_in[self.mem_weight_num].T) + T.dot(h, self.W_mem_res_hid[self.mem_weight_num].T) + self.b_mem_res[self.mem_weight_num])
        _h = T.tanh(T.dot(x, self.W_mem_hid_in[self.mem_weight_num].T) + r * T.dot(h, self.W_mem_hid_hid[self.mem_weight_num].T) + self.b_mem_hid[self.mem_weight_num])
        #ht =  g * h + (1. - g) * _h
        g = T.concatenate([g,] * self.dim, axis=1)
        ht =  g * _h + (1. - g) * h     #swapped version from paper that converges better for some reason
        return ht

    def episode_update(c_t, prev_m, q_q, W_t, b):
        m = T.nnet.relu(W_t[T.concatenate([prev_m, c_t, q_q])]+b)
        return m 
    
    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper: 
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(x, W_upd_in.T) + T.dot(h, W_upd_hid) + b_upd)
        r = T.nnet.sigmoid(T.dot(x, W_res_in.T) + T.dot(h, W_res_hid) + b_res)
        _h = T.tanh(T.dot(x, W_hid_in.T) + r * T.dot(h, W_hid_hid) + b_hid)
        #return z * h + (1. - z) * _h
        return z * _h + (1. - z) * h    #swapped version from paper that converges better for some reason
    
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    def new_episode(self, mem):
        tmp = theano.shared(np.zeros([self.batch_size, 1], dtype='float32'))
        z, z_updates = theano.scan(fn=self.episode_compute_z,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(tmp))

        g, g_updates = theano.scan(fn=self.episode_compute_g,
            sequences=z,
            non_sequences=z,)
            
        #'''
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        #'''  

        self.attentions.append(g)

        e, e_updates = theano.scan(fn=self.episode_attend,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
        return e[-1] 
    
    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )
    
    
    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            return len(self.train_input)
        elif (mode == 'test'):
            return len(self.test_input)
        else:
            raise Exception("unknown mode")
   
    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        combined = zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask)
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = zip(*combined)    
    
    def step(self, batch_idx, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            theano_fn = self.train_fn 
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
            qinfo = self.train_qinfo
        elif mode == "train_val":    
            theano_fn = self.test_fn 
            inputs = self.train_val_input
            qs = self.train_val_q
            answers = self.train_val_answer
            input_masks = self.test_input_mask
            qinfo = self.train_val_qinfo
        elif mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
            qinfo = self.test_qinfo
        else:
            raise Exception("Invalid mode")
            
        num_ma_opts = answers.shape[1]

        p_q = np.zeros((len(batch_idx), 300), dtype='float32')                                      # question input vector
        target = np.zeros((len(batch_idx)))                                                         # answer (as a single number)
        p_inp = np.zeros((len(batch_idx), self.max_sent_len, self.sent_vector_size), dtype='float32')  # story statements
        p_ans = np.zeros((len(batch_idx), num_ma_opts, 300), dtype='float32')                       # multiple choice answers
        #b_qinfo = []
        input_mask = input_masks
        for b, bi in enumerate(batch_idx):
            inp = inputs[qinfo[bi]['qid']]
            q = qs[bi]
            ans = answers[bi]
            target[b] = qinfo[bi]['correct_option']
            for i in range(len(inp)):
                p_inp[b][i] = inp[i]
            for j in range(len(ans)):
                p_ans[b][j] = self.pos_encodings(ans[j])
            p_q[b] = self.pos_encodings(q)
            #b_qinfo.append(qinfo[bi])

        ret = theano_fn(p_inp, p_q, p_ans, target)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": np.array(ret[0]),
                "answers": np.array(target),
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                "inp": np.array([inp]),
                "q" : np.array([q]),
                "probabilities": np.array([ret[0]]),
                "attentions": np.array([ret[2]]),
                }
                            
    def predict(self, data):
        # data is an array of objects like {"Q": "question", "C": "sentence ."}
        data[0]["A"] = "."
        print "==> predicting:", data
        inputs, questions, answers, input_masks = self._process_input(data)
        probabilities, loss, attentions = self.test_fn(inputs[0], questions[0], answers[0], input_masks[0])
        return probabilities, attentions

    def create_vocabulary(self, QAs, stories, v2i, w2v_vocab=None, word_thresh=1):
        """Create the vocabulary by taking all words in stories, questions, and answers taken together.
        Also, keep only words that appear in the word2vec model vocabulary (if provided with one).
        """

        print "Creating vocabulary.",
        if w2v_vocab is not None:
            print "Adding words based on word2vec"
        else:
            print "Adding all words"
        # Get all story words
        all_words = [word for story in stories for sent in story for word in sent]

        # Parse QAs to get actual words
        QA_words = []
        for QA in QAs:
            QA_words.append({})
            QA_words[-1]['q_w'] = utils.normalize_alphanumeric(QA.question.lower()).split(' ')
            QA_words[-1]['a_w'] = [utils.normalize_alphanumeric(answer.lower()).split(' ') for answer in QA.answers]

        # Append question and answer words to all_words
        for QAw in QA_words:
            all_words.extend(QAw['q_w'])
            for answer in QAw['a_w']:
                all_words.extend(answer)

        # threshold vocabulary, at least N instances of every word
        vocab = Counter(all_words)
        vocab = [k for k in vocab.keys() if vocab[k] >= word_thresh]

        # create vocabulary index
        for w in vocab:
            if w not in v2i.keys():
                if w2v_vocab is None:
                    # if word2vec is not provided, just dump the word to vocab
                    v2i[w] = len(v2i)
                elif w2v_vocab is not None and w in w2v_vocab:
                    # check if word in vocab, or else ignore
                    v2i[w] = len(v2i)

        print "Created a vocabulary of %d words. Threshold removed %.2f %% words" \
                %(len(v2i), 100*(1. * len(set(all_words)) - len(v2i))/len(all_words))

        return QA_words, v2i

    def data_in_matrix_form(self, stories, QA_words, v2i):
        """Make the QA data set compatible for memory networks by
        converting to matrix format (index into LUT vocabulary).
        """

        def add_word_or_UNK():
            if v2i.has_key(word):
                return v2i[word]
            else:
                return v2i['UNK']

        # Encode stories
        max_sentences = max([len(story) for story in stories.values()])
        max_words = max([len(sent) for story in stories.values() for sent in story])

        storyM = {}
        for imdb_key, story in stories.iteritems():
            storyM[imdb_key] = np.zeros((max_sentences, max_words), dtype='int32')
            for jj, sentence in enumerate(story):
                for kk, word in enumerate(sentence):
                    storyM[imdb_key][jj, kk] = add_word_or_UNK()

        print "#stories:", len(storyM)
        print "storyM shape (movie 1):", storyM.values()[0].shape

        # Encode questions
        max_words = max([len(qa['q_w']) for qa in QA_words])
        questionM = np.zeros((len(QA_words), max_words), dtype='int32')
        for ii, qa in enumerate(QA_words):
            for jj, word in enumerate(qa['q_w']):
                questionM[ii, jj] = add_word_or_UNK()
        print "questionM:", questionM.shape

        # Encode answers
        max_answers = max([len(qa['a_w']) for qa in QA_words])
        max_words = max([len(a) for qa in QA_words for a in qa['a_w']])
        answerM = np.zeros((len(QA_words), max_answers, max_words), dtype='int32')
        for ii, qa in enumerate(QA_words):
            for jj, answer in enumerate(qa['a_w']):
                if answer == ['']:  # if answer is empty, add an 'UNK', since every answer option should have at least one valid word
                    answerM[ii, jj, 0] = 1
                    continue
                for kk, word in enumerate(answer):
                    answerM[ii, jj, kk] = add_word_or_UNK()
        print "answerM:", answerM.shape

        return storyM, questionM, answerM

    def associate_additional_QA_info(self, QAs):
        """Get some information about the questions like story index and correct option.
        """

        qinfo = []
        for QA in QAs:
            qinfo.append({'qid':QA.qid,
                          'movie':QA.imdb_key,
                          'correct_option':QA.correct_index})
        return qinfo
