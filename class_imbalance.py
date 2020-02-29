# Dealing with class imbalance
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# use the data in the file "data_imbalance.txt" for our analysis


# Load input data
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate input data into two classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Visualize input data using scatter plot
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

# Split data into training and testing datasets
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)


'''
Next, we define the parameters for the Extremely Random Forest classifier. Note that there
is an input parameter called balance that controls whether or not we want to
algorithmically account for class imbalance. If so, then we need to add another parameter
called class_weight that tells the classifier that it should balance the weight, so that it's
proportional to the number of data points in each class:
'''
# Extremely Random Forests classifier
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if len(sys.argv) > 1:
	if sys.argv[1] == 'balance':
		params = {'n_estimators': 100, 'max_depth': 4, 'random_state':0, 'class_weight': 'balanced'}
	else:
		raise TypeError("Invalid input argument; should be 'balance'")


# Build, train, and visualize the classifier using training data:
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

# Predict the output for test dataset and visualize the output:
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')


# Compute the performance of the classifier and print the classification report:
# Evaluate classifier performance
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train),
target_names=class_names))
print("#"*40 + "\n")
print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")
plt.show()


# to execute -  python3 class_imbalance.py 
# to execute - python3 --W ignore class_imbalance.py 
# to execute - python3 class_imbalance.py balance



