from data import Dataset, Labels
from utils import evaluate
import os, sys
import math

K = 5

class KNN:
	def __init__(self):
		# bag of words document vectors
		self.bow = []

	def train(self, ds):
		"""
		ds: list of (id, x, y) where id corresponds to document file name,
		x is a string for the email document and y is the label.

		TODO: Save all the documents in the train dataset (ds) in self.bow.
		You need to transform the documents into vector space before saving
		in self.bow.
		"""
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

		#print(self.bow)

		pass

	def predict(self, x):
		"""
		x: string of words in the document.

		TODO: Predict class for x.
		1. Transform x to vector space.
		2. Find k nearest neighbors.
		3. Return the class which is most common in the neighbors.
		"""
		def sumVectorSquares(vectorToSum):
			summedVector = 0.0
			for value in vectorToSum:
				#print(vectorToSum[i])
				summedVector += value**2
			return summedVector

		#Multiple each value in two vectors and add
		def sumProdOfTwoVectors(vector1, vector2):
			#print(vector1)
			#print(vector2)
			product = 0.0
			for i in range(len(vector1)):
				product += vector1[i]*vector2[i]
			return product

		def takeSecond(elem):
			return elem[1]

		vectorIndex = 0
		xBowVector = []
		xBowVector.append([0]*(len(self.bow[0])))
		for word in x.lower().split(): 
			if word in self.bow[0]:
				vectorIndex = self.bow[0].index(word)
			#print(xBowVector)
				xBowVector[0][vectorIndex] += 1
		del xBowVector[0][0]
		del xBowVector[0][0]

		#cosine similarity = A . B / |A| * |B|
		#find 3 closes documents (K) should be 3 or 5
		index = 0
		cosineSimilarities = []
		bowVectorIndex = 2
		origLabel = 0
		skip = 0
		product = 0.0
		AtimesB = 0.0
		tempList3 = []

		for vector in self.bow:
			if skip > 1:

				tempList3 = vector[2]

				productOfVectorAndxBow = sumProdOfTwoVectors(xBowVector[0], tempList3)
					#origLabel = self.bow[index+1].pop()
				A = math.sqrt(sumVectorSquares(xBowVector[0]))

				B = math.sqrt(sumVectorSquares(tempList3)) #used to be index + 2
					#vectorLabel = self.bow[1][index+2]

				product = productOfVectorAndxBow
				AtimesB = A*B
					#cosineSimilarities[(vector, xBowVector, vectorLabel)] = productOfVectorAndxBow/(A*B)
				#cosineSimilarities[vector[1]] = product/AtimesB
				cosineSimilarities.append([vector[0], product/AtimesB])

				index += 1
				bowVectorIndex += 1

			skip +=1
		cosineSimilarities.sort(key=takeSecond, reverse=True)

		topK = {}
		K = 5
		for i in range(K):
			if cosineSimilarities[i][0] not in topK:
				topK[cosineSimilarities[i][0]] = 1
			else:
				topK[cosineSimilarities[i][0]] += 1

		#print(topK)
		sortedTopK = sorted(topK.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

		return Labels(sortedTopK[0][0])

def main(train_split):
	knn = KNN()
	ds = Dataset(train_split).fetch()
	val_ds = Dataset('val').fetch()
	knn.train(ds)

	# Evaluate the trained model on training data set.
	print('-'*20 + ' TRAIN ' + '-'*20)
	evaluate(knn, ds)
	# Evaluate the trained model on validation data set.
	print('-'*20 + ' VAL ' + '-'*20)
	evaluate(knn, val_ds)

	# students should ignore this part.
	# test dataset is not public.
	# only used by the grader.
	if 'GRADING' in os.environ:
		print('\n' + '-'*20 + ' TEST ' + '-'*20)
		test_ds = Dataset('test').fetch()
		evaluate(knn, test_ds)


if __name__ == "__main__":
	train_split = 'train'
	if len(sys.argv) > 1 and sys.argv[1] == 'train_half':
		train_split = 'train_half'
	main(train_split)
