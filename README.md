# csce-470-pa2
Assignment to classify documents using 3 different classifying algorithms: Naive Bayes, KNN, and Rocchio

The starter code contains two directories: src and data.

The src directory contains the following files:
  1. knn.py: Implementation of the k-nearest neighbor classifier. (3 points)
  2. naive-bayes.py: Implementation of the naive bayes classifier. (3 points)
  3. rocchio.py: Implementation of the rocchio classifier. (3 points)
  4. utils.py: Helper functions used in the classifiers including the evaluation function. 5. data.py: Responsible for reading the train, validation 
     and test split.

The data directory includes the email documents in the train directory. The labels for each email is provided in train-split.txt 
(contains labels for 357 training documents), train-half-split.txt (contains labels for half of the training documents) and val-split.txt 
(contains labels for validation documents).
You are expected to modify the classifier files only and not change utils.py and data.py, but you should read all the python files to best understand 
how to optimally implement the classification algorithms. Each classifier contains train() and predict() functions that contain helpful TODOs to be 
completed by you. Read the code at the end of the classifier files to see how these functions are used to train and evaluate the models. Feel free to add 
print() statements for debugging or visualizing the arguments for the functions but remove all print() statements before your final submission.
The classifiers can be trained and evaluated by running the corresponding files with python3 and will not run with any version of python(<3).

# By default all classifiers will train on the entire training
# set, train-split.txt
$ python3 naive-bayes.py
$ python3 knn.py
$ python3 rocchio.py
# For using train-half-split.txt for training, pass
# “train_half” as an argument (works for all classifiers): $ python3 naive-bayes.py train_half

The above classifier scripts will print the confusion matrix, accuracy, precision, recall and F1 score for train and validation data split.
