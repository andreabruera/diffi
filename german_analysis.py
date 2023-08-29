import gensim
import numpy
import os
import scipy

from gensim.models import Word2Vec
from scipy import spatial

### reading w2v
w2v = Word2Vec.load(os.path.join(
                         'models',
                         'word2vec_de_opensubs_sdewac_param-mandera2017',
                         'word2vec_de_opensubs_sdewac_param-mandera2017.model',
                         )
                    )

### reading 2011 dataset
indices = {
           '_2011_' :
           {
               0 : 'word',
               7 : 'aoa',
           },
           '_2016_' :
           {
               0 : 'word',
               10 : 'aoa',
           }
           }

dataset = {v : list() for v in ['word', 'aoa']}

for year, mapper in indices.items():
    for f in os.listdir('german_norms'):
        if year not in f:
            continue
        if year == '_2011_':
            header_len = 2
            delimiter=';'
        else:
            header_len = 1
            delimiter='\t'
        counter = 0
        with open(os.path.join('german_norms', f), encoding='utf-8') as i:
            for l in i:
                if counter < header_len:
                    counter += 1
                    continue
                line = l.strip().split(delimiter)
                if line[0] == '':
                    continue
                #print(line)
                for idx, var in mapper.items():
                    val = line[idx] if var=='word' else (float(line[idx].replace(',', '.'))-1) / (7 - 1)
                    dataset[var].append(val)

### preparing the dataset - although some values are double, so they need to be averaged
word_values = {k : list() for k in dataset['word']}

for k, v in zip(dataset['word'], dataset['aoa']):
    word_values[k].append(v)
for k, v in word_values.items():
    try:
        assert k in w2v.wv.key_to_index.keys()
    except AssertionError:
        print(k)
    word_values[k] = [numpy.average(v)]
import pdb; pdb.set_trace()
