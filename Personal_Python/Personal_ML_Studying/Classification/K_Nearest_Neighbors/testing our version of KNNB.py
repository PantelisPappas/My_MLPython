import numpy as np
from math import sqrt
import warnings
import pandas as pd
import random
from collections import Counter

def k_nearest_neightbors(data, predict, k=200):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!....idiot')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) #using the in-built nunpy euclidean distance function
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / float(k)

    return vote_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True) #replacing ? for -99999 to know that's an outlier. We don't want strings in our algorithm
df.drop(['id'], 1 , inplace=True) #dropping 'id' since it's useless in clasifying data
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set  =  {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[:-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neightbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1
print('Accuracy:', float(correct)/float(total))
