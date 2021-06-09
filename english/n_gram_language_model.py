from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

model = defaultdict(lambda:defaultdict(lambda : 0))


for sent in reuters.sents():
	for w1,w2,w3 in trigrams(sent, pad_right = True, pad_left = True):
		model[(w1,w2)][w3] += 1


# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count


print(dict(model["I","am"]))


import random

# starting words
text = ["I", "am"]
sentence_finished = False
 
while not sentence_finished:
  # select a random probability threshold  
  r = random.random()
  accumulator = .0

  for word in model[tuple(text[-2:])].keys():
      accumulator += model[tuple(text[-2:])][word]
      # select words that are above the probability threshold
      if accumulator >= r:
          text.append(word)
          break

  if text[-2:] == [None, None]:
      sentence_finished = True
 
print (' '.join([t for t in text if t]))