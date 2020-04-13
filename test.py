import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

bankdata = pd.read_csv("rezultaaaat_1.csv")
print(bankdata.shape)  # number of lines and columns
print(bankdata.head())  # first 4 lines
"""Data processing -> divide data into attributes and labels
                      divide in training and data sets"""
attributes = bankdata.drop('Class', axis=1)  # will contain all columns except Class which is the label
labels = bankdata['Class']  # label
print(attributes)
print(labels)
# dividing data
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.20)  # 80% of data will be training, 20% test

svclassifier = SVC(kernel='sigmoid',random_state=0)  # letting the algorithm know that he s working with linear data
svclassifier.fit(X_train, y_train)  # actual training

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(y_test)
print(y_pred)