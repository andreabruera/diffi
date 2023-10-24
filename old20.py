import numpy
import os
import pickle
import re

from tqdm import tqdm

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def multiprocessing_levenshtein(inputs):
    w = inputs[0]
    other_words = inputs[1]
    levs = list()
    for w_two in other_words:
        levs.append(levenshtein(w, w_two))
    return (w, levs)

'''
### reading 2011 dataset
indices = {
           '_2011_' :
           {
               0 : 'word',
           },
           '_2016_' :
           {
               0 : 'word',
           }
           }

dataset = {v : list() for v in ['word']}

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

relevant_words = list(set(dataset['word']))
#print(relevant_words)
### reading phil's dataset
relevant_words = list()
with open(os.path.join('data', 'phil_clean.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            rel_idx = line.index('Words')
            continue
        relevant_words.append(line[rel_idx])
'''
### reading phil's candidate dataset
relevant_words = list()
with open(os.path.join('data', 'german_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            continue
        relevant_words.append(line[0])

with open(os.path.join('pickles', 'sdewac_lemma_freqs.pkl'), 'rb') as i:
    all_sdewac_freqs = pickle.load(i)

all_sdewac_freqs = {k : v for k, v in all_sdewac_freqs.items() if len(k)>1 and len(re.findall('\||@|<|>',k))==0}
### number of words in the original OLD20 paper
max_n = 35502
other_words = sorted(all_sdewac_freqs.items(), key=lambda item : item[1], reverse=True)[:max_n]
#print(other_words[-1])

def print_stuff(inputs):
    print(inputs)

old20_scores = {w : 0 for w in relevant_words}
for w in tqdm(relevant_words):
    _, lev_vals = multiprocessing_levenshtein([w, other_words])
    score = numpy.average(sorted(lev_vals, reverse=True)[:20])
    #print([w, score])
    old20_scores[w] = score

#with open('old20_scores.tsv', 'w') as o:
with open('old20_scores_candidate_nouns_phil.tsv', 'w') as o:
    o.write('word\told20 score (based on the top {} lemmas in sdewac)\n'.format(max_n))
    for k, v in old20_scores.items():
        o.write('{}\t{}\n'.format(k, v))
