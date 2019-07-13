#!usr/bin/env python
# -*- coding:utf-8 -*-


import pandas as pd
import numpy as np
from scipy import stats
import json,math

from sklearn.preprocessing import MinMaxScaler, Imputer


def cos(vector1,vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a,b in zip(vector1,vector2):
        dot_product += a*b
        normA += a**2
        normB += b**2
    if normA == 0.0 or normB==0.0:
        return 0
    else:
        return dot_product / ((normA*normB)**0.5)


def Euclidean(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    distance = np.sqrt(np.sum(np.square(vector1 - vector2)))
    return -math.log(distance)




data = pd.read_csv('MTURK-771.csv', header=None)
wordsList = np.array(data.iloc[:, [0, 1]])
simScore = np.array(data.iloc[:, [2]])

we_fd = open('word_embedding.txt','r')
word2vec = json.load(we_fd)


predScoreList = np.zeros((len(simScore), 1))
for i, (word1, word2) in enumerate(wordsList):
    print "process #%d words pair [%s,%s]" % (i, word1, word2)
    predScoreList[i, 0] = cos(word2vec[word1],word2vec[word2])
    print predScoreList[i, 0]

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
impList = imp.fit_transform(predScoreList)
mms = MinMaxScaler(feature_range=(1.0, 5.0))
impMmsList = mms.fit_transform(impList)

print '\n'*2
(pearson_cor, pvalue) = stats.pearsonr(simScore, impMmsList)
print 'pearson_correlation=', pearson_cor[0], '   pvalue=',pvalue[0]
(spearman_cor, pvalue) = stats.spearmanr(simScore, impMmsList)
print 'spearman_correlation=', spearman_cor, '   pvalue=',pvalue

submitData = np.hstack((wordsList, simScore, impMmsList))
(pd.DataFrame(submitData)).to_csv('embedding_relateness.csv', index=False,
                                  header=["Word1", "Word2", "gt_relateness", "pred_relateness"])

we_fd.close()