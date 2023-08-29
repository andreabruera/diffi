import gensim
import numpy
import os
import pickle
import scipy

from gensim.models import Word2Vec
from scipy import spatial


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
                    val = line[idx].strip() if var=='word' else (float(line[idx].replace(',', '.'))-1) / (7 - 1)
                    dataset[var].append(val)

final_dataset = dict()

### preparing the dataset - although some values are double, so they need to be averaged
word_aoa = {k : list() for k in dataset['word']}

for k, v in zip(dataset['word'], dataset['aoa']):
    word_aoa[k].append(v)
del dataset
for k, v in word_aoa.items():
    word_aoa[k] = numpy.average(v)

### z-scoring
mean_aoa = numpy.average(list(word_aoa.values()))
std_aoa = numpy.std(list(word_aoa.values()))
mean_length = numpy.average([len(k) for k in word_aoa.keys()])
std_length = numpy.std([len(k) for k in word_aoa.keys()])
final_dataset['word'] = list()
final_dataset['word_length'] = list()
final_dataset['aoa'] = list()
for k, v in word_aoa.items():
    final_dataset['word'].append(k)
    final_dataset['word_length'].append((len(k)-mean_length)/std_length)
    final_dataset['aoa'].append((v-mean_aoa)/std_aoa)

### reading pickles
with open(os.path.join('pickles', 'opensubs_word_freqs.pkl'), 'rb') as i:
    all_opensubs_freqs = pickle.load(i)
relevant_opensubs_freqs = dict()
for w in word_aoa.keys():
    opensubs_w = w.replace('-', '')
    try:
        relevant_opensubs_freqs[w] = numpy.log10(all_opensubs_freqs[opensubs_w])
        #relevant_opensubs_freqs[w] = all_opensubs_freqs[opensubs_w]
    except KeyError:
        relevant_opensubs_freqs[w] = numpy.nan
print('total missing words in opensubs: {}'.format(len([w for w, v in relevant_opensubs_freqs.items() if str(v)=='nan'])))
print('missing words in opensubs: {}'.format([w for w, v in relevant_opensubs_freqs.items() if str(v)=='nan']))

### z-scoring
mean_os_freq = numpy.nanmean(list(relevant_opensubs_freqs.values()))
std_os_freq = numpy.nanstd(list(relevant_opensubs_freqs.values()))
final_dataset['log10_freq_opensubs'] = list()
for k in final_dataset['word']:
    v = relevant_opensubs_freqs[k]
    if v != numpy.nan:
        v = (v-mean_os_freq)/std_os_freq
    final_dataset['log10_freq_opensubs'].append(v)

with open(os.path.join('pickles', 'sdewac_word_freqs.pkl'), 'rb') as i:
    all_sdewac_freqs = pickle.load(i)
relevant_sdewac_freqs = dict()
for w in word_aoa.keys():
    try:
        relevant_sdewac_freqs[w] = numpy.log10(all_sdewac_freqs[w])
        #relevant_sdewac_freqs[w] = all_sdewac_freqs[w]
    except KeyError:
        relevant_sdewac_freqs[w] = numpy.nan
print('total missing words in sdewac: {}'.format(len([w for w, v in relevant_sdewac_freqs.items() if str(v)=='nan'])))
print('missing words in sdewac: {}'.format([w for w, v in relevant_sdewac_freqs.items() if str(v)=='nan']))

### z-scoring
mean_sdewac_freq = numpy.nanmean(list(relevant_sdewac_freqs.values()))
std_sdewac_freq = numpy.nanstd(list(relevant_sdewac_freqs.values()))
final_dataset['log10_freq_sdewac'] = list()
for k in final_dataset['word']:
    v = relevant_sdewac_freqs[k]
    if v != numpy.nan:
        v = (v-mean_sdewac_freq)/std_sdewac_freq
    final_dataset['log10_freq_sdewac'].append(v)

### reading concreteness
all_concreteness_scores = dict()
with open(os.path.join('german_norms', 'affective_norms.txt')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        all_concreteness_scores[line[0]] = float(line[1])
relevant_concreteness_scores = dict()
for w in word_aoa.keys():
    conc_w = w.replace('ß', 'ss')
    try:
        relevant_concreteness_scores[w] = all_concreteness_scores[conc_w]
    except KeyError:
        relevant_concreteness_scores[w] = numpy.nan

### z-scoring
mean_conc = numpy.nanmean(list(relevant_concreteness_scores.values()))
std_conc = numpy.nanstd(list(relevant_concreteness_scores.values()))
final_dataset['concreteness'] = list()
for k in final_dataset['word']:
    v = relevant_concreteness_scores[k]
    if v != numpy.nan:
        v = (v-mean_conc)/std_conc
    final_dataset['concreteness'].append(v)

print('total missing words in concreteness ratings: {}'.format(len([w for w, v in relevant_concreteness_scores.items() if str(v)=='nan'])))
print('missing words in concreteness ratings: {}'.format([w for w, v in relevant_concreteness_scores.items() if str(v)=='nan']))

### reading old20
all_old20_scores = dict()
with open(os.path.join('german_norms', 'old20_scores.tsv')) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        all_old20_scores[line[0]] = float(line[1])

relevant_old20_scores = dict()
for w in word_aoa.keys():
    try:
        relevant_old20_scores[w] = all_old20_scores[w]
    except KeyError:
        relevant_old20_scores[w] = numpy.nan

print('total missing words in old20 scores: {}'.format(len([w for w, v in relevant_old20_scores.items() if str(v)=='nan'])))
print('missing words in old20 scores: {}'.format([w for w, v in relevant_old20_scores.items() if str(v)=='nan']))

### z-scoring
mean_old20 = numpy.nanmean(list(relevant_old20_scores.values()))
std_old20 = numpy.nanstd(list(relevant_old20_scores.values()))
final_dataset['OLD20'] = list()
for k in final_dataset['word']:
    v = relevant_old20_scores[k]
    if v != numpy.nan:
        v = (v-mean_old20)/std_old20
    final_dataset['OLD20'].append(v)

cols = [
        'word',
        'word_length',
        'aoa',
        'OLD20',
        'log10_freq_opensubs',
        'log10_freq_sdewac',
        'concreteness',
        ]
with open('german_stimuli_zscored_values.tsv', 'w') as o:
    o.write('\t'.join(cols))
    o.write('\n')
    for i in range(len(final_dataset['word'])):
        for col in cols:
            o.write('{}\t'.format(final_dataset[col][i]))
        o.write('\n')
