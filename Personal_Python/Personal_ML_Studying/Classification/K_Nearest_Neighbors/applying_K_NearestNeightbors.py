import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #replacing ? for -99999 to know that's an outlier. We don't want strings in our algorithm
df.drop(['id'], 1 , inplace=True) #dropping 'id' since it's useless in clasifying data

X = np.array(df.drop(['class'] , 1) )     #features
y = np.array(df['class'])                 #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,3,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)
print prediction
