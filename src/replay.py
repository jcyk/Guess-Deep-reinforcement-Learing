import random

class Replay(object):
	def __init__(self,Q,minibatch_size,replay_size):
		self.Q_network = Q
		self.size = replay_size
		self.minibatch_size = minibatch_size
		self.gue_pool = []
		self.des_pool = []
		self.count = 0
		self.update_count = 0

	def des_add(self,conversation,target_word,description,guess):
		self.des_pool.append((conversation,target_word,description,guess))

	def gue_add(self,target_word,conversation,description,guess,_description):
		self.count+=1
		self.gue_pool.append((target_word,conversation,description,guess,_description))
		if self.count > self.size and self.count%self.minibatch_size == 0:
			self.gue_pool = self.gue_pool[:self.size]
			self.des_pool = self.des_pool[:self.size]
			self.update_count+=1
			if self.update_count %2 ==0:
				self.Q_network.update_guesser(random.sample(self.gue_pool,self.minibatch_size))
			else:
				self.Q_network.update_describler(random.sample(self.des_pool,self.minibatch_size))

