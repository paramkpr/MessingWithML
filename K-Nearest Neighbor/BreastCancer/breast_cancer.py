import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

df = (pd.read_csv('breast-cancer-wisconsin.data.txt'))
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy, 'is the accuracy of this algorithm.')

# noinspection PyPep8
example_values = np.array([4,1,1,1,1,2,1,2,1])
example_values = example_values.reshape(1, -1)

prediction = clf.predict(example_values)
print(prediction, '(2 for benign, 4 for malignant)')
