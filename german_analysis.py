import numpy
import os
import scipy

from scipy import spatial

### reading w2v
print('now reading w2v...')
w2v = dict()
with open(os.path.join('models', 'dewac_cbow.txt'), encoding='utf-8') as i:
    for l in i:
        line = l.strip().split()
        word = line[0][1:-1]
        vector = numpy.array(line[1:], dtype=numpy.float64)
        print(line)
        assert vector.shape == (400, )
        w2v[word] = vector
print('loaded!')
import pdb; pdb.set_trace()

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
            delimiter=','
        counter = 0
        with open(os.path.join('german_norms', f), encoding='latin-1') as i:
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
    word_values[k] = [numpy.average(v)]
    if k.lower() not in w2v.keys():
        print(k)
