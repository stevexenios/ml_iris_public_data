#################################################################
#
#  ML - HW 1
#
#  Martine De Cock
#  Steve G. Mwangi

#################################################################

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score


# Read the dataset into a dataframe and map the labels to numbers
df = pd.read_csv('iris.csv')
map_to_int = {'setosa':0, 'versicolor':1, 'virginica':2}
df["label"] = df["species"].replace(map_to_int)
#print(df)

# Separate the input features from the label
features = list(df.columns[:4])
X = df[features]
y = df["label"]

# Train a decision tree and compute its training accuracy
clf = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy')
results = cross_val_score(clf, X, y, cv=10, scoring='accuracy') # 1 of 2 Added this line
print("10 Fold Cross validation values: \n", results)
print("\nMean of Values = ", np.mean(results)) # 2 of 2 Added this line
print("\nMean of Values = ", results.mean()) # 2 of 2 Added this line
clf.fit(X, y)
print("Training Accuracy = ", metrics.accuracy_score(y,clf.predict(X)))