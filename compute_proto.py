import os
import numpy
import pickle
import scipy

from scipy import spatial

nouns_candidates = list()
trans_de = dict()
with open(os.path.join('data', 'german_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            continue
        nouns_candidates.append(line[0])
        trans_de[line[0]] = line[1]
### loading aligned german fasttext
ft_de_file = os.path.join('pickles', 'ft_de_aligned.pkl')
if os.path.exists(ft_de_file):
    with open(ft_de_file, 'rb') as i:
        ft_de = pickle.load(i)
### reading phil's dataset
centroid_words = {
        'animate' : list(),
        'natural' : list(),
        'inanimate' : list(),
        'innatural' : list(),
    }
with open(os.path.join('data', 'phil_clean.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            header = line.copy()
            continue
        if float(line[header.index('Animate')]) == 1:
            centroid_words['animate'].append(line[header.index('Words')])
        if float(line[header.index('Animate')]) == 0:
            centroid_words['inanimate'].append(line[header.index('Words')])
        if float(line[header.index('Natural')]) == 1:
            centroid_words['natural'].append(line[header.index('Words')])
        if float(line[header.index('Natural')]) == 0:
            centroid_words['innatural'].append(line[header.index('Words')])
centroid_vectors = {k : numpy.average([ft_de[w.lower()] for w in v if w.lower() in ft_de.keys()], axis=0) for k,v in centroid_words.items()}

keyz = list(centroid_vectors.keys()).copy()
with open(os.path.join('data', 'prototypicality_german_nouns_phil.tsv'), 'w') as o:
    o.write('word\t')
    for k in keyz:
        o.write('{}_prototypicality\t'.format(k))
    o.write('\n')
    for w in nouns_candidates:
        o.write('{}\t'.format(w))
        vec = ft_de[w.lower()]
        for k in keyz:
            sim = 1 - scipy.spatial.distance.cosine(vec, centroid_vectors[k])
            o.write('{}\t'.format(sim))
        o.write('\n')

