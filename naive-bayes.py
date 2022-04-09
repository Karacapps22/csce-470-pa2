from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class NaiveBayes:
	def __init__(self):
		# total number of documents in the training set.
		self.n_doc_total = 0
		# total number of documents for each label/class in the trainin set.
		self.n_doc = {l: 0 for l in Labels}
		# frequency of words for each label in the trainng set.
		self.vocab = {l: {} for l in Labels}

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.
		
		TODO: Loop over the dataset (ds) and update self.n_doc_total,
		self.n_doc and self.vocab.
		"""
		freqWordsForLabel = 0
		ListOfFreqWordsForLabel = []
		numDocsForLabel = ""
		for l in Labels:
			self.n_doc[l] = 0
		

		for i in range(len(ds)):
			#getting numDocsForLabel: check label

			#loop through and count the number of times the label appears in each doccument
			listOfDocWords = ds[i][1].lower().split()

			self.n_doc[ds[i][2]] += 1
			for j in listOfDocWords:
				if j in self.vocab[ds[i][2]]:
					self.vocab[ds[i][2]][j] += 1
				else:
					self.vocab[ds[i][2]][j] = 1
			self.n_doc_total += 1




	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Use self.n_doc_total, self.n_doc and self.vocab to calculate the
		prior and likelihood probabilities.
		Add the log of prior and likelihood probabilities.
		Use MAP estimation to return the Label with hight score as
		the predicted label.
		"""

		def wordProb(self, word, labelNum):
			totalNumWordsForClass_countC = 0
			countWC = 0

			if word in self.vocab[labelNum]:
				countWC = self.vocab[labelNum][word]
			totalFreqOfWordsForAllClasses_V = 0
			wordProbability = 0

			for l in Labels:
				if word in self.vocab[l]:
					totalFreqOfWordsForAllClasses_V += self.vocab[l][word]
			
			for key in self.vocab.keys():
				if key in self.vocab[labelNum]:
					totalNumWordsForClass_countC += self.vocab[labelNum][key]
			wordProbability = (countWC + 1)/(totalNumWordsForClass_countC + totalFreqOfWordsForAllClasses_V +1)
			return wordProbability

		priorProbs = {}
		words = x.lower().split()

		for key in Labels:
			priorProbs[key] = self.n_doc[key]/self.n_doc_total
			
		prediction = 0.0
		finalPrediction = 0.0
		maxLabel = 0
		for l in Labels:
			prediction = math.log(priorProbs[l])
			for word in words:
				prediction += math.log(wordProb(self, word, l))
			prediction = 10**prediction
			if prediction >= finalPrediction:
				#print("Predictions vs finalPrediction ")
				#print(prediction)
				#print(finalPrediction)
				#print(l)
				finalPrediction = prediction
				maxLabel = l

		return Labels(maxLabel)


def main(train_split):
	nb = NaiveBayes()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	nb.train(ds)
	
	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(nb, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(nb, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(nb, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
