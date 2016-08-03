import random
import os
import numpy as np

class Data_loader(object):

	def __init__(self,embedding_file,embedding_size):
		path = os.path.join(embedding_file,str(embedding_size))
		self.vocab,raw_data = [],[]
		with open(path) as f:
			for line in f.readlines():
				data = line.split()
				assert len(data) == embedding_size+1
				self.vocab.append(data[0])
				raw_data.append([float(d) for d in data[1:]])
		self.vocab.append('IMG1')
		self.vocab.append('IMG2')
		raw_data.append(raw_data[0])
		raw_data.append(raw_data[0])
		self.vocab_size = len(self.vocab)
		self.embeddings = np.asarray(raw_data,dtype=np.float32)
		
	def get_target_word(self):
		return random.randint(0,self.vocab_size-1)
		