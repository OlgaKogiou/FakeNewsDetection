from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from FeatureExtraction import *
import json
import numpy as np
import csv
import string
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt


# Function to split dataset to train-test subset
def split_data(data, labels):
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(data):
        return data[train_index], data[test_index], labels[train_index], labels[test_index]


# Function to train Naive Bayes Classifier
def train_NB(train_data, train_labels):
    model = MultinomialNB().fit(train_data, train_labels)
    y_predict_for_train_data = model.predict(train_data)
    average_accuracy = np.mean(train_labels == y_predict_for_train_data) * 100
    print("The average training accuracy is {}%".format(average_accuracy))
    return model


# Function to train Random Forest Classifier
def train_random_forest(train_data, train_labels, est):
    model = RandomForestClassifier(n_estimators=est).fit(train_data, train_labels)
    y_predict_for_train_data = model.predict(train_data)
    average_accuracy = np.mean(train_labels == y_predict_for_train_data) * 100
    print("The average training accuracy is {}%".format(average_accuracy))
    return model


# Function to train Support Vector Machine
def train_SVC(train_data, train_labels, k, c):
    model = svm.SVC(C=c, kernel=k).fit(train_data, train_labels)
    y_predict_for_train_data = model.predict(train_data)
    average_accuracy = np.mean(train_labels == y_predict_for_train_data) * 100
    print("The average training accuracy is {}".format(average_accuracy))
    return model


# Function to test classifiers using the fixed models
def test_classifier(clf, validate_data, validate_labels):
    predicted = clf.predict(validate_data)
    average_accuracy = np.mean(validate_labels == predicted) * 100
    print("The average testing accuracy is {}".format(average_accuracy))
    return predicted


# dataset reading
def read_datasets():
    # Read the datasets
    fake = pd.read_csv("/home/tanagno/Desktop/Fake.csv")
    true = pd.read_csv("/home/tanagno/Desktop/True.csv")

    # Printing the data
    return fake, true


def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str


# dataset cleaning wrapper
def preprocess(text):
    text = text.str.replace('"', '')
    text = text.str.replace('-', '')
    text = text.str.replace(':', '')
    return text


# cleaning data from characters and adding/ removing columns
def data_cleaning(fake, true):
    fake['label'] = '1'
    true['label'] = '0'

    data = pd.concat([fake, true]).reset_index(drop=True)

    data = shuffle(data)
    data = data.reset_index(drop=True)

    data.drop(['date'], axis=1, inplace=True)
    data['text'] = preprocess(data['text'])
    return data


# Function to plot confusion-matrix
# (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
def plot_confusion_matrix(cm, file_name, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)
