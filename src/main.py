import os
import sys

import numpy as np
import tensorflow as tf 
from DQL import DQL
from Q_network import Q_network
from replay import Replay
from data_loader import Data_loader

import pprint
pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_string('embedding_file','../glove.6B','where is the embedding file ')
flags.DEFINE_integer('embedding_size',100,'embedding size [100]')
flags.DEFINE_float('step_size',0.1,'step size [0.1]')
flags.DEFINE_float('greedy_ratio',0.8,'greedy ratio [0.8]')
flags.DEFINE_integer('target_frequency',32,'update target network every [32] steps')
flags.DEFINE_integer('hidden_units',128,'number of hidden units [128]')
flags.DEFINE_integer('final_units',128,'number of final units [128]')
flags.DEFINE_integer('minibatch_size',128,'minibatch size [128]')
flags.DEFINE_integer('replay_size',20000,'replay size [20000]')
flags.DEFINE_integer('budget',10000,'budget [10000]')
FLAGS = flags.FLAGS

def main(_):
	pp.pprint(flags.FLAGS.__flags)
	with tf.Session() as sess:
		data_loader = Data_loader(FLAGS.embedding_file,FLAGS.embedding_size)
		q_network = Q_network(sess,FLAGS.embedding_size,FLAGS.step_size,FLAGS.target_frequency,FLAGS.hidden_units,FLAGS.final_units,FLAGS.greedy_ratio,data_loader)
		replay = Replay(q_network,FLAGS.minibatch_size,FLAGS.replay_size)
		model = DQL(FLAGS.budget,data_loader,q_network,replay)
		model.run()

if __name__ == '__main__':
	tf.app.run(main)