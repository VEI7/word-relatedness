#!usr/bin/env python
# -*- coding:utf-8 -*-

from optparse import OptionParser
from nltk.corpus import wordnet as wn

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.preprocessing import MinMaxScaler, Imputer

parser = OptionParser("Help for wordnet_relateness algorithm",
        description="word pair relateness calculation algorithm implemented in python.",
        version="1.0"
    )
parser.add_option("-i", "--input", action="store", dest="input",type="string", help="Input file")
parser.add_option("-o", "--output", action="store", dest="output",type="string", help="Output file")
parser.add_option("-m", "--similarity_method", action="store", dest="sm",type="string",help="select method to calculate relateness,include path,lch,wup")

options, args = parser.parse_args()


data = pd.read_csv(options.input, header=None)
wordsList = np.array(data.iloc[:, [0, 1]])
simScore = np.array(data.iloc[:, [2]])

predScoreList = np.zeros((len(simScore), 1))
for i, (word1, word2) in enumerate(wordsList):
    print "process #%d words pair [%s,%s]" % (i, word1, word2)
    count = 0
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            if options.sm == 'lch':
                if synset1.pos() != synset2.pos():
                    continue
                score = synset1.lch_similarity(synset2)
            elif options.sm == 'path':
                score = synset1.path_similarity(synset2)
            elif options.sm == 'wup':
                score = synset1.wup_similarity(synset2)
            if score is not None:
                predScoreList[i, 0] += score
                count += 1
                print synset1, "similarity", synset2, "is ", score
            else:
                print synset1, "similarity", synset2, "is None"
    predScoreList[i, 0] = predScoreList[i, 0] * 1.0 / count



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
(pd.DataFrame(submitData)).to_csv(options.output, index=False,
                                  header=["Word1", "Word2", "gt_relateness", "pred_relateness"])