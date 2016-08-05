import numpy as np
import tensorflow as tf
import random
from tensorflow.python.ops import rnn,rnn_cell
from Model import Model
from util import contextual_rnn, Contextual_GRUCell

class Q_network(Model):
    def __init__(self,sess,embedding_size,step_size,target_frequency,hidden_units,final_units,greedy_ratio,data_loader):
        self.sess = sess
        self.embedding_size = embedding_size
        self.step_size = step_size
        self.hidden_units = hidden_units
        self.data_loader = data_loader
        self.final_units = final_units
        self.greedy_ratio = greedy_ratio
        self.target_frequency = target_frequency
        self.update_count = 0
        self.vocab_size = data_loader.vocab_size
        with tf.variable_scope('online'):
            self.update_guesser_op,self.update_describler_op,self.guesser_value,self.describer_value,self.target_v,self.i_know,self.conv_d,self.conv_g,self.conv_len,self.last_d,self.target_w,self._guess_,self._description_= self.build_model()
           #update_guesser_op      #update_describler_op    #guesser_value      #describer_value      #target_v #i_know        #conversation_d #conversation_g#conversation_len#last_description#target_word#_guess_#_description_
        num_variables =  len(tf.trainable_variables())
        with tf.variable_scope('target'):
            _1,_2,self.t_guesser_value,self.t_describer_value,_3,self.t_i_know,self.t_conv_d,self.t_conv_g,self.t_conv_len,self.t_last_d,self.t_target_w,self.t_guess_,self.t_description_= self.build_model()
        
        self.update_target = []
        trainable_variables = tf.trainable_variables()
        for i in xrange(num_variables):
            self.update_target.append(trainable_variables[num_variables + i].assign(trainable_variables[i]))
        tf.initialize_all_variables().run()
        self.reset_target()

    def reset_target(self):
        self.sess.run(self.update_target)

    def build_model(self):
        target_v = tf.placeholder(tf.float32,[None])
        conversation_d = tf.placeholder(tf.int32,[None,None])
        conversation_g = tf.placeholder(tf.int32,[None,None])
        conversation_len = tf.placeholder(tf.int32,[None])
        last_description = tf.placeholder(tf.int32,[None])
        target_word = tf.placeholder(tf.int32,[None])
        i_know = tf.placeholder(tf.int32,[None])
        
        word_embeddings = tf.get_variable('word_embed',initializer = tf.convert_to_tensor(self.data_loader.embeddings,dtype=tf.float32),trainable=False)
        conv_d = tf.nn.embedding_lookup(word_embeddings,conversation_d)
        conv_g = tf.nn.embedding_lookup(word_embeddings,conversation_g)
        know = tf.nn.embedding_lookup(word_embeddings,i_know) 
        conv = tf.concat(2,[conv_g,conv_d])

        with tf.variable_scope('guesser'):
            last_des = tf.nn.embedding_lookup(word_embeddings,last_description)
            gue_cell = rnn_cell.GRUCell(self.hidden_units)
            _, gue_state = rnn.dynamic_rnn(gue_cell,conv,conversation_len,dtype=tf.float32)
            gue_repr = tf.tanh(rnn_cell._linear([gue_state,last_des],self.final_units,True))
            gue_core = tf.get_variable('gue_core',[self.final_units,self.embedding_size])
            gue_ready = tf.matmul(gue_repr,gue_core)
            guesser_value = tf.reduce_sum(tf.mul(gue_ready,know),1)
            gue_pred = tf.matmul(gue_ready,word_embeddings,transpose_b=True)
            _guess_ = tf.nn.top_k(gue_pred,self.vocab_size)
        
        with tf.variable_scope('describer'):
            target = tf.nn.embedding_lookup(word_embeddings,target_word)
            des_cell = Contextual_GRUCell(self.hidden_units)
            _, des_state = contextual_rnn(des_cell,conv,target,conversation_len,dtype=tf.float32)
            des_repr = tf.tanh(rnn_cell._linear([des_state,target],self.final_units,True))
            des_core = tf.get_variable('des_core',[self.final_units,self.embedding_size])
            des_ready = tf.matmul(des_repr,des_core)
            describer_value = tf.reduce_sum(tf.mul(des_ready,know),1)
            des_pred = tf.matmul(des_ready,word_embeddings,transpose_b=True)
            _description_ = tf.nn.top_k(des_pred,self.vocab_size)

        optimizer = tf.train.GradientDescentOptimizer(self.step_size)
        update_guesser_op = optimizer.minimize(tf.reduce_sum(tf.square(target_v-guesser_value)))
        update_describler_op = optimizer.minimize(tf.reduce_sum(tf.square(target_v-describer_value)))
        return update_guesser_op,update_describler_op,guesser_value,describer_value,target_v,i_know,conversation_d,conversation_g,conversation_len,last_description,target_word,_guess_,_description_

    def split(self,conversation):        
        if len(conversation)==0 or type(conversation[0]) is not list:
            conversation = [conversation]
        conversation_d,conversation_g,conversation_len = [],[],[]
        for conv in conversation:
            ds,gs = [],[]
            for idx,x in enumerate(conv):
                if idx%2==0:
                    ds.append(x)
                else:
                    gs.append(x)
            conversation_d.append(ds)
            conversation_g.append(gs) 
            conversation_len.append(len(ds))      
        return conversation_d,conversation_g,conversation_len

    def f_guesser(self,conversation,description):
        conversation_d,conversation_g,conversation_len = self.split(conversation)
        value,idx = self.sess.run(self._guess_,
                                    feed_dict={self.conv_d:self.to_tensor(conversation_d),
                                               self.conv_g:self.to_tensor(conversation_g),
                                               self.conv_len: self.to_tensor(conversation_len),
                                               self.last_d:self.to_tensor(description)})
        if type(description) is not list:
            return value,idx,[set(conversation+[description])]

        ban = [ set(x+[y]) if conversation<self.vocab_size-1 else set([]) for x,y in zip(conversation,description)]
        return value,idx,ban

    def guesser_sample(self,conversation,description):
        if random.random()<self.greedy_ratio:
            value,word = self.guess(conversation,description)
            return word[0]
        ban = set(conversation+[description])
        res = random.randint(0,self.vocab_size-1)
        while True:
            if res not in ban:
                return res
            res+=1
            if res == self.vocab_size:
                res = 0
    
    def guess(self,conversation,description):
        value,idx,ban = self.f_guesser(conversation,description)
        res_v,res_w = [],[]
        for v,x,y in zip(list(value),list(idx),ban):
            for i,j in reversed(zip(x,v)):
                if i not in y:
                    res_v.append(j)
                    res_w.append(i)
                    break
        return res_v,res_w

    def f_describler(self,conversation,target_word):
        conversation_d,conversation_g,conversation_len = self.split(conversation)
        value,idx = self.sess.run(self._description_,
                                    feed_dict={self.conv_d:self.to_tensor(conversation_d),
                                               self.conv_g:self.to_tensor(conversation_g),
                                               self.conv_len: self.to_tensor(conversation_len),
                                               self.target_w:self.to_tensor(target_word)})
        if type(target_word) is not list:
            return value,idx,[set(conversation+[target_word])]
        ban = [ set(x+[y]) if conversation<self.vocab_size-1 else set([]) for x,y in zip(conversation,target_word)]
        return value,idx,ban  

    def describler_sample(self,conversation,target_word):
        if random.random()<self.greedy_ratio:
            value,word = self.describle(conversation,target_word)
            return word[0]
        ban = set(conversation+[target_word])
        res = random.randint(0,self.vocab_size-1)
        while True:
            if res not in ban:
                return res
            res+=1
            if res == self.vocab_size:
                res = 0

    def describle(self,conversation,target_word):
        value,idx,ban= self.f_describler(conversation,target_word)
        res_v,res_w = [],[]
        for v,x,y in zip(list(value),list(idx),ban):
            for i,j in reversed(zip(x,v)):
                if i not in y:
                    res_v.append(j)
                    res_w.append(i)
                    break
        return res_v,res_w

    def to_tensor(self,x):
        if type(x) is not list:
            return [x]
        if type(x[0]) is list:
            lens = [len(i) for i in x]
            max_len = max(lens)
            values = np.zeros(shape=(len(x),max_len),dtype=np.int32)
            for idx,i in enumerate(x):
                values[idx][:lens[idx]] = i
            return values
        return np.asarray(x,dtype=np.int32)

    def update_guesser(self,minibatch):
        self.update_count+=1
        minibatch_conv,minibatch_d = [],[]
        for _t,_conv,_d1,_g,_d2 in minibatch:
            minibatch_conv.append(_conv+[_d1,_g])
            minibatch_d.append(_d2)
        
        _,g = self.guess(minibatch_conv,minibatch_d)
        conv_d,conv_g,conv_len = self.split(minibatch_conv)
        v = self.sess.run(self.t_guesser_value,
                                    feed_dict={self.t_conv_d:self.to_tensor(conv_d),
                                               self.t_conv_g:self.to_tensor(conv_g),
                                               self.t_conv_len:self.to_tensor(conv_len),
                                               self.t_last_d:self.to_tensor(minibatch_d),
                                               self.t_i_know:self.to_tensor(g)
                                    })
        minibatch_conv,minibatch_d,minibatch_g = [],[],[]
        idx = 0
        for _t,_conv,_d1,_g,_d2 in minibatch:
            minibatch_conv.append(_conv)
            minibatch_d.append(_d1)
            minibatch_g.append(_g)
            if _t == _g:
                v[idx] = 0
            idx+=1
        conv_d,conv_g,conv_len = self.split(minibatch_conv)
        _ = self.sess.run(self.update_guesser_op,
                                    feed_dict={self.conv_d:self.to_tensor(conv_d),
                                               self.conv_g:self.to_tensor(conv_g),
                                               self.conv_len:self.to_tensor(conv_len),
                                               self.last_d:self.to_tensor(minibatch_d),
                                               self.i_know:self.to_tensor(minibatch_g),
                                               self.target_v:v+1
                                    })
        if self.update_count%2==0 and (self.update_count/2)%self.target_frequency==0:
            self.reset_target()

    def update_describler(self,minibatch):
        self.update_count+=1
        minibatch_conv,minibatch_t =[],[]
        for _conv,_t,_d,_g in minibatch:
            minibatch_conv.append(_conv+[_d,_g])
            minibatch_t.append(_t)
        
        _,d = self.describle(minibatch_conv,minibatch_t)
        conv_d,conv_g,conv_len = self.split(minibatch_conv)
        v = self.sess.run(self.t_describer_value,
                                    feed_dict={self.t_conv_d:self.to_tensor(conv_d),
                                               self.t_conv_g:self.to_tensor(conv_g),
                                               self.t_conv_len:self.to_tensor(conv_len),
                                               self.t_target_w:self.to_tensor(minibatch_t),
                                               self.t_i_know:self.to_tensor(d)
                                    })
        minibatch_conv,minibatch_d = [x[0] for x in minibatch],[]
        idx = 0
        for  _conv,_t,_d,_g in minibatch:
            minibatch_d.append(_d)
            if _t == _g:
                v[idx] = 0
            idx+=1
        conv_d,conv_g,conv_len = self.split(minibatch_conv)
        _ = self.sess.run(self.update_describler_op,
                                    feed_dict={self.conv_d:self.to_tensor(conv_d),
                                               self.conv_g:self.to_tensor(conv_g),
                                               self.conv_len:self.to_tensor(conv_len),
                                               self.target_w:self.to_tensor(minibatch_t),
                                               self.i_know:self.to_tensor(minibatch_d),
                                               self.target_v:v+1
                                    })
        if self.update_count%2==0 and (self.update_count/2)%self.target_frequency==0:
            self.reset_target()