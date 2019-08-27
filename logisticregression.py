import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import matplotlib.pyplot as plt
import time

class LogisticRegression:

    # Set the regularization weight Lambda
    def __init__(self, Lambda=5):
        self.Lambda = Lambda

    # Compute the Sigmoid function of input a
    def sigmoid(self, a):
        if a > 0:
            return 1 / (1 + math.exp(-a))
        # For a < 0, expression is changed to avoid numerical issue
        else:
            return math.exp(a) / (math.exp(a) + 1)

    # Compute the gradient of the Negative Log Likelihood (Loss function) with regularization w.r.t. current weight w
    def gradient(self):
        g = np.zeros((1, self.train_data.shape[1]))
        for i in range(self.train_data.shape[0]):
            g += (self.sigmoid(np.matmul(self.train_data[i], np.transpose(self.w))) - self.train_label[i]) * \
                 self.train_data[i]
        # Add the gradient of regularization to the end
        return g + self.Lambda * self.w

    # Compute the Hessian matrix with regularization w.r.t. current weight w
    def hessian(self):
        r = [self.sigmoid(np.matmul(x, np.transpose(self.w))) * (1 - self.sigmoid(np.matmul(x, np.transpose(self.w))))
             for x in self.train_data]
        R = np.diag(r)
        h = np.matmul(np.matmul(np.transpose(self.train_data), R), self.train_data) + self.Lambda * np.identity(
            self.train_data.shape[1])
        return h

    # Preprocess the raw dataframes for easy computation
    def preprocess(self, data, label):
        # Apply Min_Max normalization (for faster convergence)
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        # Add ones for the bias term w0
        X_0 = np.ones((data.shape[0], 1))
        data = np.concatenate((X_0, data), axis=1)
        # convert the class label to {1, 0} (for computing gradient)
        bin_label = [1 if y == self.c1 else 0 for y in label]
        label = np.array(bin_label).reshape(-1, 1)
        return data, label

    # Compute the optimal w iteratively by Newton's Method
    def fit(self, train_data, train_label, c1):
        # Define c1 as class label 1, the other as 0
        self.c1 = c1
        self.train_data, self.train_label = self.preprocess(train_data.values, train_label.values)
        # Randomly initialize the weight vector
        self.w = np.random.rand(1, self.train_data.shape[1]) / 100
        for i in range(10):
            g = self.gradient()
            h = self.hessian()
            self.w = self.w - np.matmul(g, np.linalg.inv(h))

    # Evaluate the classification performance for all data points and for each class
    def accuracy(self, test_data, test_label):
        # Apply the same transformation for testing dataframes
        test_data, test_label = self.preprocess(test_data.values, test_label.values)
        # Classify x to class label 1 if xw_t is positive, 0 otherwise
        pred = np.array([1 if np.matmul(i, np.transpose(self.w)) > 0 else 0 for i in test_data])
        # Generate the confusion matrix
        cm = confusion_matrix(test_label, pred.reshape(-1, 1))
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        total_score = 100 * (cm.astype('float') / cm.sum()).diagonal().sum()
        # Class 1 Accuracy (Precision) = TP / (TP + FP)
        # Class 2 Accuracy = TN / (FN + TN)
        individual_score = 100 * (cm.astype('float') / cm.sum(axis=1)).diagonal()
        return total_score, individual_score

# 10-Fold Cross Validation for choosing the optimal Lambda from
# [0, .2, ,4, .6, .8, 1, 2, 4, 6, 8, 10, 50, 100, 150, 200, 250, 300, 350]
def cross_validation(k_fold=10):
    accuracy_per_Lambda = []
    Lambdas = [i*0.2 for i in range(6)] + [i*2 for i in range(1,6)] + [i*50 for i in range(1,8)]
    for i in Lambdas:
        accuracy_per_run = []
        for j in range(k_fold):
            df_validate_data = pd.read_csv('dataset/trainData' + str(j + 1) + '.csv', header=None)
            df_validate_label = pd.read_csv('dataset/trainLabels' + str(j + 1) + '.csv', header=None)
            df_train_data, df_train_label = merge_train_files(k_fold, skip=j)
            # Create a logistic regression classifier
            clf = LogisticRegression(Lambda=i)
            clf.fit(df_train_data, df_train_label, 5)
            accuracy_per_run.append(clf.accuracy(df_validate_data, df_validate_label)[0])
            # At the end of each k-fold cv, calculate the average accuracy
            if j == k_fold - 1:
                avg_accuracy = np.mean(np.array(accuracy_per_run))
                accuracy_per_Lambda.append(avg_accuracy)
                print('Lambda = {:5.1f}, Accuracy = {:4.1f}%'.format(i, avg_accuracy))
    # Find the optimal Lambda which yields maximum average accuracy
    optimal_Lambda = Lambdas[np.argmax(np.array(accuracy_per_Lambda))]
    print('The best Lambda = ', optimal_Lambda)
    return optimal_Lambda, Lambdas, accuracy_per_Lambda

# Merge multiple csv file into one data and one label data frames. Optionally, we can exclude certain files
def merge_train_files(num_of_files, skip=None):
    df_train_data = pd.DataFrame()
    df_train_label = pd.DataFrame()
    for k in range(num_of_files):
        if k == skip:
            continue
        data = pd.read_csv('dataset/trainData' + str(k + 1) + '.csv', header=None)
        df_train_data = df_train_data.append(data, ignore_index=True)
        label = pd.read_csv('dataset/trainLabels' + str(k + 1) + '.csv', header=None)
        df_train_label = df_train_label.append(label, ignore_index=True)
    return df_train_data, df_train_label

# Find the optimal Lambda, also x (all Lambdas) and y (accuracies) for plotting
optimal_Lambda, x, y = cross_validation()

# Merge 10 training data and label files
df_train_data, df_train_label = merge_train_files(10)
# Read test data and label files
df_test_data = pd.read_csv('dataset/testData.csv', header=None)
df_test_label = pd.read_csv('dataset/testLabels.csv', header=None)

running_times = []
print('Run 100 times: Approximate wait time: 23 sec')
for i in range(100):
    start_time = time.time()
    # Train the MoG classifier with training dataframes and the Lambda from K-Fold Cross Validation
    clf = LogisticRegression(Lambda=10)
    clf.fit(df_train_data, df_train_label, 5)
    total_score, individual_score = clf.accuracy(df_test_data, df_test_label)
    running_times.append(time.time() - start_time)
    if i == 99:
        print('\nTotal Accuracy: {:5.2f}%\nClass 1: \'5\', Accuracy: {:5.2f}%\nClass 0: \'6\', Accuracy: {:5.2f}%'.format(
            total_score, individual_score[1], individual_score[0]))

print('\nAverage running time: {:6.4f}s'.format(np.array(running_times).mean()))

# Try logistic regression from sklearn library and output its score
clf = linear_model.LogisticRegression(random_state=0, solver='newton-cg').fit(df_train_data, df_train_label.values.reshape(-1,))
print('\nTotal Accuracy(sklearn): {}%'.format(100 * clf.score(df_train_data, df_train_label)))

# Plot the relationship between Lambda and accuracy
plt.plot(x, y)
plt.xlabel('Lambda', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.title('10-Fold Cross Validation: Lambda vs Accuracy', fontsize=18)
plt.show()