import re
import os
embed = []
vocab = []
with open('/Users/jcyk/tensorflow/guess/glove.6B/glove.6B.100d.txt') as f:
	for line in f.readlines():
		data = line.split()
		assert len(data) == 101
		vocab.append(data[0])
		embed.append(' '.join(data[1:]))

stopwords = []
with open('/Users/jcyk/tensorflow/guess/stopwords/english') as f:
	for line in f.readlines():
		w = line.strip()
		if w.endswith("\'t") or w.endswith("\'s"):
			stopwords.append(w[:-2])
		else:
			stopwords.append(w)

count = 0
stopwords = set(stopwords)
with open('removed','wb') as f:
	for idx,w in enumerate(vocab):
		if w.isalpha() and w not in stopwords:
			f.write(w+' '+embed[idx]+'\n')
			count+=1
			if count >= 300:
				break
print count

#os.system('java -cp ".:stanford-corenlp.jar" Main removed')


#sett = []
#with open('stemmed') as f:
#	for line in f.readlines():
#		w = line.strip()
#		sett.append(w)
#print len(set(sett))

