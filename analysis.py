import gensim
import itertools
import matplotlib
import nltk
import numpy
import os
import sklearn

from gensim import downloader
from matplotlib import pyplot
from nltk.corpus import wordnet

### read norms

folder = 'norms'
assert len(os.listdir(folder)) == 3
norms = dict()
pos = dict()

for f in os.listdir(folder):
    counter = 0
    if 'Concreteness' in f:
        key = 'concreteness'
        rel_idxs = [0, 2]
    elif 'AoA' in f:
        key = 'AoA'
        rel_idxs = [0, 3]
    else:
        continue
    norms[key] = dict()
    with open(os.path.join(folder, f), errors='ignore') as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            if len(line) < 3:
                continue
            try:
                norms[key][line[rel_idxs[0]]] = float(line[rel_idxs[1]])
            except ValueError:
                continue
            if key == 'concreteness':
                pos[line[0]] = line[-1]

### read wilson & al data

folder = 'data'
assert len(os.listdir(folder)) == 2
couples_dict = dict()
for f in os.listdir(folder):
    key = f.split('.')[0]
    couples_dict[key] = list()
    counter = 0
    with open(os.path.join(folder, f)) as i:
        for l in i:
            if counter == 0:
                counter += 1
                continue
            ### corrections for w2v
            for old, new in [('doughnut', 'donut'), ('pretence', 'pretense'), ('axe', 'ax')]:
                l = l.replace('{}\t'.format(old), '{}\t'.format(new))
            line = l.strip().split('\t')
            if key == 'matches':
                rel_idxs = [1, 2, 5]
            else:
                rel_idxs = [0, 2, 4]
            #for idx in rel_idxs[:2]:
            #    try:
            #        wv[line[idx]]
            #    except KeyError:
            #        print(line[idx])
            couples_dict[key].append([line[rel_idxs[0]], line[rel_idxs[1]], float(line[rel_idxs[2]])])
words_total = set([w for lst in couples_dict.values() for case in lst for w in case[:2]])
words_per_cat = {k : set([w for case in lst for w in case[:2]]) for k, lst in couples_dict.items()}

### loading w2v
wv = gensim.downloader.load('word2vec-google-news-300')
#sims_per_cat = {k : [wv.similarity(case[0], case[1]) for case in lst] for k, lst in couples_dict.items()}

### printouts!

print('\n')
print('total words: {}'.format(len(words_total)))
for k, v in words_per_cat.items():
    print('words in {}: {}'.format(k, len(v)))
#for k, v in sims_per_cat.items():
#    print('avg sim for {}: {}'.format(k, numpy.average(v)))
#    print('avg std for {}: {}'.format(k, numpy.std(v)))
for k, tuples in couples_dict.items():
    #print('\n')
    ### ratings
    for key, current_variable in norms.items():
        print('\n')
        missing_words = [w for w in words_per_cat[k] if w not in current_variable.keys()]
        #print('missing words for {}: {}'.format(key, missing_words))
        values = list()
        for tup in tuples:
            if tup[0] in missing_words or tup[1] in missing_words:
                continue 
            diff = abs(current_variable[tup[0]]-current_variable[tup[1]])
            values.append(diff)
        print('avg difference in {} for {}: {}'.format(key, k, numpy.average(values)))
        print('std of differences in {} for {}: {}'.format(key, k, numpy.std(values)))
    ### similarities
    ### w2v
    print('\n')
    w2v_sims = [wv.similarity(tup[0], tup[1]) for tup in tuples]
    print('avg w2v similarity for {}: {}'.format(k, numpy.average(w2v_sims)))
    print('std of w2v similarity for {}: {}'.format(k, numpy.std(w2v_sims)))
    ### wordnet
    wn_sims = list()
    diff_senses = list()
    print('\n')
    for tup in tuples:
        if tup[0] in missing_words or tup[1] in missing_words:
            continue 
        words = list()
        tup_sims = list()
        for w in tup[:2]:
            #current_pos = pos[w]
            w_syn = wordnet.synsets(w)
            words.append(w_syn)
        diff_senses.append(abs(len(words[0])-len(words[1])))
        combs = list(itertools.product(words[0], words[1]))
        for c in combs:
            #sim = wordnet.path_similarity(c[0], c[1])
            sim = wordnet.wup_similarity(c[0], c[1])
            tup_sims.append(sim)
        wn_sims.append(numpy.average(tup_sims))
    print('avg wordnet similarity for {}: {}'.format(k, numpy.average(wn_sims)))
    print('std of wordnet similarity for {}: {}'.format(k, numpy.std(wn_sims)))
    print('avg differences in senses for {}: {}'.format(k, numpy.average(diff_senses)))
    print('std of differences in senses for {}: {}'.format(k, numpy.std(diff_senses)))
