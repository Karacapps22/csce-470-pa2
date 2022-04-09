from data import Dataset, Labels
from utils import evaluate
import math
import os, sys


class Rocchio:
	def __init__(self):
		# centroids vectors for each Label in the training set.
		self.centroids = {l: {} for l in Labels}
		self.bow = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Loop over all the samples in the training set, convert the
		documents to vectors and find the centroid for each Label.
		"""
		#1.convert training docs into raw term frequency vectors 
		#2.normalize document vectors using Euclidean normalization
		#3. calculates centroid for each class using normalized training document vectors
	

		def sumVectorSquares(vectorToSum):
			summedVector = 0.0
			for value in vectorToSum:
				#print(vectorToSum[i])
				summedVector += value**2
			return summedVector

		#Multiple each value in two vectors and add
		def sumProdOfTwoVectors(vector1, vector2):
			product = 0.0
			for i in range(len(vector1)):
				product += vector1[i]*vector2[i]
			return product
		dictOfAllWords = {}
		index = 0
		for doc in ds:
			for word in doc[1].lower().split():
				if word not in dictOfAllWords:
					dictOfAllWords[word] = index
					index += 1


		tempList = [""]
		tempList.append("")
		for word in dictOfAllWords:
			tempList.append(word)
		self.bow.append(tempList)


		vectorIndex = 0
		docIndex = 0
		value = 0
		ELengths = []
		for doc in ds:
			tempList2 = []
			tempList2.append(doc[2])
			tempList2.append(doc[0])
			tempList2.append([0]*(len(dictOfAllWords)+100))

			for word in doc[1].lower().split():
				#find word and put value into vector
				word = word.lower()
				if word in self.bow[0]:
					vectorIndex = self.bow[0].index(word)
					tempList2[2][self.bow[0].index(word)] = tempList2[2][self.bow[0].index(word)] + 1
			docIndex += 1
			self.bow.append(tempList2)

			E = 0.0
			for i in range(len(tempList2[2])):
				E += (tempList2[2][i])**2
			E = math.sqrt(E)

			#if doc[2] not in ELengths:
			#	ELengths[doc[2]] = E
			#else:
			#	ELengths[doc[2]] += E
			ELengths.append([doc[2], doc[0], E])


			#self.centroids = {l: {} for l in Labels}
		print ("Elengths", ELengths)
		for l in Labels:
			u = 0.0
			docCount = 0
			print ("Label:",l)
			for doc in ELengths:
				print("doc:",doc)
				if doc[0] is l:
					u += doc[2]
					docCount += 1
					print("Adding E ", doc[2], " to u",u)

			uC =  (1/docCount)*(u)
			self.centroids[l] = uC




	def predict(self, x):
		"""
		x: string of words in the document.
		
		TODO: Convert x to vector, find the closest centroid and return the
		label corresponding to the closest centroid.
		"""
		#given new doc to predict 1. convert doc into raw term frequency vector
		#2. normalize document vector using Euclidean normalization
		#3. calculate cosine similarities between normalized new doc vector and class centroid vectors
		#4. assign class label of closest centroid to given doc

		return Labels(0)

def main(train_split):
	rocchio = Rocchio()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	rocchio.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(rocchio, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(rocchio, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(rocchio, test_ds)

if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
