from replay import Replay
from copy import copy

class DQL(object):
	def __init__(self,budget,data_loader,Q,replay):
		self.budget = budget
		self.data_loader = data_loader
		self.Q = Q
		self.replay = replay
		
	def run(self):
		for i in xrange(self.budget):
			target_word = self.data_loader.get_target_word()
			#print 'start',self.data_loader.vocab[target_word]
			step_count = 0
			conversation = []
			while True:
				description = self.Q.describler_sample(copy(conversation),target_word)
				
				_guess = self.Q.guess(copy(conversation),description)[1][0]
				self.replay.des_add(copy(conversation),target_word,description,_guess)
				
				guess = self.Q.guesser_sample(copy(conversation),description)
				
				_description = self.Q.describle(copy(conversation),target_word)[1][0]
				self.replay.gue_add(target_word,copy(conversation),description,guess,_description)
				
				conversation.extend([description,guess])

				step_count+=1
				#print step_count,self.data_loader.vocab[description],self.data_loader.vocab[guess]
				if guess == target_word:
					print step_count,'end'
					break

