import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import warnings
from collections import Counter
style.use('fivethirtyeight')

dataSet = {'k': [[1, 2], [2, 3], [3, 1]], 'r':[[6, 5], [7, 7], [8, 6], [7, 8]]}
new_features = [3, 3]

for i in dataSet:
    for j in dataSet[i]:
        plt.scatter(j[0], j[1], s=100, color=i)


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("There are more voting arguments than k's value.")

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])

    votes = []
    for n in sorted(distances)[:k]:
        votes.append(n[1])

    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


result = k_nearest_neighbors(dataSet, new_features)
print(result)

plt.scatter(new_features[0], new_features[1], s=100, marker='>', color=result, label='Grouped Feature')
plt.legend(loc=4)
plt.show()
