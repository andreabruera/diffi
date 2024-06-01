import googletrans
import numpy
import os
import pickle
import scipy
import sklearn

from scipy import spatial
from sklearn import linear_model
from tqdm import tqdm

### loading lemma frequencies
with open(os.path.join('pickles', 'sdewac_lemma_pos_freqs.pkl'), 'rb') as i:
    lemma_pos = pickle.load(i)
with open(os.path.join('pickles', 'sdewac_lemma_freqs.pkl'), 'rb') as i:
    lemma_freqs = pickle.load(i)
### loading word frequencies
with open(os.path.join('pickles', 'sdewac_word_freqs.pkl'), 'rb') as i:
    word_freqs = pickle.load(i)

### loading aligned german fasttext
ft_de_file = os.path.join('pickles', 'ft_de_aligned.pkl')
if os.path.exists(ft_de_file):
    with open(ft_de_file, 'rb') as i:
        ft_de = pickle.load(i)
else:
    ft_de = dict()
    with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'de', 'wiki.de.align.vec')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split(' ')
            if l_i == 0:
                continue
            ft_de[line[0]] = numpy.array(line[1:], dtype=numpy.float64)
    with open(ft_de_file, 'wb') as i:
        pickle.dump(ft_de, i)

### frequency threshold: 1000
freq_threshold = 100
nouns_candidates = list()
for w, freq in lemma_freqs.items():
    if freq > freq_threshold:
        w_pos = sorted(lemma_pos[w].items(), key=lambda item : item[1], reverse=True)
        marker = False
        ### recognizing as nouns words with high relative dominance (>40%) of the noun usage
        if w_pos[0][0] == 'NN':
            marker = True
        else:
            if 'NN' in [p[0] for p in w_pos]:
                proportion = lemma_pos[w]['NN'] / sum([p[1] for p in w_pos])
                if proportion > 0.4:
                    marker = True
        if marker:
            if w.lower() in ft_de.keys():
                nouns_candidates.append(w)

trans_de = dict()
with open(os.path.join('data', 'nouns_phil_semantic_norms.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            continue
        trans_de[line[0]] = line[1]

### learning a transformation of tf_de words
translator = googletrans.Translator()
for w in tqdm(nouns_candidates):
    if w in trans_de.keys():
        continue
    trans = translator.translate(w, src='de', dest='en').text.lower()
    trans_de[w] = trans

with open(os.path.join('data', 'german_nouns_phil_100.tsv'), 'w') as o:
    o.write('word\ten_google_translation\n')
    for de, en in trans_de.items():
        o.write('{}\t{}\n'.format(de, en))
