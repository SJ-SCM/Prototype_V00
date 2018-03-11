from __future__ import division  # only for Python 2

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import os
import psutil

def getData():
    f = open('hiq_v4.fa', "r")
    lines = f.readlines()
    f.close()
    raw_label = []
    label = []  # label is a 1d array of (sub)species with indexes matching sequences in features list
    old_features = []
    features = []  # features is a 1d array of sequences with indexes matching (sub)species in label list
    temp = []
    data = []  # data is a 2d array of sequence and (sub)species
    for i in lines:

        if not '>' in i:
            temp.append(i)
        else:
            raw_label.append(i)
            old_features.append(temp)
            temp = []
    old_features.append(temp)
    temp = []

    del old_features[0]

    # print 'converting the 2d array: '
    string = ''
    for i in old_features:
        for j in i:
            string += j
            if string[-1:] == '\n':
                string = string[:-1]
        features.append(string)
        string = ''

    # print '==================== PROCESSING RAW LABEL ==========================='

    for i in raw_label:
        head, sep, tail = i.partition('sp=')
        tail = tail[:-2]
        label.append(tail)

    for i in range(len(label)):
        data.append([features[i], label[i]])
    target = label

    target = set(target)  # target is a set of all (sub)species i.e. no duplicated

    return data

def vectorize_data(data):
    vectorized_genome = []
    list = []
    vectorized_data = []
    label = []
    for i in data:

        for j in i[0]:
            if j == 'A':
                list.append([1,0,0,0])
            elif j == 'G':
                list.append([0,1,0,0])
            elif j == 'C':
                list.append([0,0,1,0])
            elif j == 'T':
                list.append([0,0,0,1])
        vectorized_genome.append(list)
        list = []
        label.append(i[1])

    for i in range(len(vectorized_genome)):
        vectorized_data.append([vectorized_genome[i],label[i]])

    return vectorized_data

def print_section(vectorized_data,x,y): #prints all list indexes in an specific interval of the given array
    count = 0
    for i in vectorized_data:
        if count >= x and count < y:
            print('count: ', count, i)
        count += 1

def check_vector(data, vectorized_data): # this function tests the vectorized data by comparing to the normal data
    string = ''
    test_vector = []
    for i in vectorized_data:
        for j in i[0]:
            if j == [1,0,0,0]:
                string += 'A'
            elif j == [0,1,0,0]:
                string += 'G'
            elif j == [0,0,1,0]:
                string += 'C'
            elif j == [0,0,0,1]:
                string += 'T'
        test_vector.append(string)
        string = ''
    count = 0
    test_passed = True
    for i in range(len(test_vector)):
        if test_vector[i] != data[i][0]:
            test_passed = False
            print('ERROR AT: ', i)
        else:
            count += 1
    print('number of samples', count, '\n',  'test passed: ', test_passed)

# ======== MACHINE LEARNING SECTION ======================================================

def flatten(vectorized_data):
    features = []
    for i in range(len(vectorized_data)):
        temp = reshape2d1d(vectorized_data[i][0])
        if len(temp) < 960:
            print(vectorized_data[i][1])
        temp = temp[:960]
        features.append(temp)
    return features

def reshape2d1d(sublist):
    result = []

    for i in sublist:
        result += i
    return result


def machine_learning():
    data = getData()
    vectorized_data = vectorize_data(data)
    # check_vector(data, vectorized_data)

    features_2D = []
    labels = []

    for i in vectorized_data:
        features_2D.append(i[0])
        labels.append(i[1])

    features = flatten(vectorized_data)

    # =============================================== MACHINE LEARNING ================================================


    # split the data into training and testing
    train_feats, test_feats, train_labels, test_labels = tts(features, labels, test_size=0.01)

    # SVM with RBF kernel. Default setting of SVM.
    # clf = svm.SVC(verbose=True)

    # SVM with linear kernel
    # clf = svm.SVC(kernel='linear',verbose=True,tol=0.0000001)

    # MLPClassifier
    # clf = MLPClassifier(verbose=True,tol=0.0000001)

    # GaussianProcessClassifier
    # clf = GaussianProcessClassifier()

    # Decision Tree Classifier
    # clf = tree.DecisionTreeClassifier()

    # Random Forest Classifier
    clf = RandomForestClassifier(verbose=True)

    # AdaBoostClassifier
    # clf = AdaBoostClassifier()

    # K neighbors classifier
    # clf = KNeighborsClassifier(verbose=True,tol=0.0000001)

    # print the details of the Classifier used
    print("Using", clf)

    # training
    clf.fit(train_feats, train_labels)

    # predictions
    predictions = clf.predict(test_feats)
    print("\nPredictions:", predictions)

    score = 0
    for i in range(len(predictions)):
        if predictions[i] == test_labels[i]:
            score += 1
    print("Accuracy:", (score / len(predictions)) * 100, "%")

    # or, just do this for accuracy
    print(accuracy_score(test_labels, predictions))

    # pred = clf.predict_proba(test_feats)

    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('RAM memory usage in GB:', memoryUse)
    return accuracy_score(test_labels, predictions)


def mean_accuracy():
    total_score = 0
    for i in range(20):
        total_score += machine_learning()
    print('average accuracy', total_score/20)


