# Random forest

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # Does not work -from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from utilities import visualize_classifier

'''
Define an argument parser for Python so that we can take the classifier type as an input
parameter. Depending on this parameter, we can construct a Random Forest classifier or an
Extremely Random forest classifier:
'''
# Argument parser
def build_arg_parser():
	parser = argparse.ArgumentParser(description='Classify data using Ensemble Learning techniques')
	parser.add_argument('--classifier-type', dest='classifier_type', required=True, choices=['rf', 'erf'], help="Type of 		classifier\to use; can be either 'rf' or 'erf'")
	return parser


#Define the main function and parse the input arguments:
if __name__=='__main__':
	# Parse the input arguments
	args = build_arg_parser().parse_args()
	classifier_type = args.classifier_type

	# Load input data
	input_file = 'data_random_forests.txt'
	data = np.loadtxt(input_file, delimiter=',')
	X, y = data[:, :-1], data[:, -1]

	# Separate input data into three classes based on labels
	class_0 = np.array(X[y==0])
	class_1 = np.array(X[y==1])
	class_2 = np.array(X[y==2])

	# Visualize input data
	plt.figure()
	plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',edgecolors='black', linewidth=1, marker='s')
	plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',edgecolors='black', linewidth=1, marker='o')
	plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',edgecolors='black', linewidth=1, marker='^')
	plt.title('Input data')

	# Split data into training and testing datasets
	# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

	'''
	Define the parameters to be used when we construct the classifier. The n_estimators parameter refers to the number of 		trees that will be constructed. The max_depth parameter refers to the maximum number of levels in each tree. The 		random_state parameter refers to the seed value of the random number generator needed to initialize the random forest
	classifier algorithm:
	'''
	# Ensemble Learning classifier
	params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

	if classifier_type == 'rf':
		classifier = RandomForestClassifier(**params)
	else:
		classifier = ExtraTreesClassifier(**params)

	# Train and visualize the classifier:
	classifier.fit(X_train, y_train)
	visualize_classifier(classifier, X_train, y_train, 'Training dataset')


	# Compute the output based on the test dataset and visualize it:
	y_test_pred = classifier.predict(X_test)
	visualize_classifier(classifier, X_test, y_test, 'Test dataset')


	# Evaluate classifier performance
	class_names = ['Class-0', 'Class-1', 'Class-2']
	print("\n" + "#"*40)
	print("\nClassifier performance on training dataset\n")
	print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
	print("#"*40 + "\n")
	print("#"*40)
	print("\nClassifier performance on test dataset\n")
	print(classification_report(y_test, y_test_pred, target_names=class_names))
	print("#"*40 + "\n")

# to execute - python3 random_forest.py --classifier-type rf
# to execute - python3 random_forest.py --classifier-type erf

