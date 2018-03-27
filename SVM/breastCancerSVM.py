import numpy as np
import pandas as pd
from sklearn import svm, preprocessing, model_selection

df = (pd.read_csv('breast-cancer-wisconsin.data.txt'))
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)


X = np.array(df.drop(['class'], 1)) #  Features
y = np.array(df['class']) #  Labels

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

prediction_data = np.array([9,4,6,1,8,7,2,6,3]).reshape(1, -1)
prediction = clf.predict(prediction_data)
print(prediction)
