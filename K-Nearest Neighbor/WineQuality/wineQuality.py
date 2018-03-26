import numpy as np
from sklearn import neighbors, preprocessing, model_selection
import pandas as pd

df = pd.read_csv('winequality-red.csv')
df.astype(float).values.tolist()

X = np.array(df.drop(['quality'], 1))
y = np.array(df['quality'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print(clf.score(X, y))

to_predict = np.array([7, 0.57, 0.23, 6.8, 0.09, 20, 104, 0.9978, 9.36, 0.9, 10.5]).reshape(1, -1)

print(clf.predict(to_predict))
