import pandas as pd
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing
style.use('ggplot')

df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)


def handle_non_numerical_data(dataFrame):
    columns = dataFrame.columns.values

    for column in columns:
        text_digit_val = {}

        def convert_to_int(val):
            return text_digit_val[val]

        if dataFrame[column].dtype != np.int64 and dataFrame[column].dtype != np.float64:
            column_contents = dataFrame[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_val:
                    text_digit_val[unique] = x
                    x += 1

            dataFrame[column] = list(map(convert_to_int, dataFrame[column]))

    return dataFrame


df = handle_non_numerical_data(df)
print(df.head())

clf = KMeans(n_clusters=2)

X = np.array(df.drop(['survived'], 1).astype(float))
preprocessing.scale(X)
y = np.array(df['survived'])

clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
