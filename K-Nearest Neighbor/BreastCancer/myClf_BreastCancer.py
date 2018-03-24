import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random


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


# Data Setup
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)


# Cross Validation
test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

# Testing
for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print(correct / total)
