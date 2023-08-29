import gensim
import numpy
import os
import random
import scipy

from gensim.models import Word2Vec
from scipy import spatial
from tqdm import tqdm

### reading w2v
w2v = Word2Vec.load(os.path.join(
                         'models',
                         'word2vec_de_opensubs+sdewac_param-mandera2017',
                         'word2vec_de_opensubs+sdewac_param-mandera2017.model',
                         )
                    )

### first we load the data
stimuli = dict()
with open('german_stimuli_zscored_values.tsv') as i:
    counter = 0
    for l in i:
        line = l.strip().split('\t')
        if counter == 0:
            header = line[1:]
            counter += 1
            continue
        if 'nan' not in line and line[0] in w2v.wv.key_to_index.keys():
            if line[0].capitalize() == line[0]:
                stimuli[line[0]] = numpy.array(line[1:], dtype=numpy.float32)
                assert stimuli[line[0]].shape == (len(header), )

### then we compute the word difficulty
### length + aoa + old20 - log10_freq_opensubs - log10_freq_sdewac - concreteness
formula_vector = numpy.zeros(shape=(len(header), ))
formula_vector[header.index('word_length')] = 1
formula_vector[header.index('aoa')] = 2
formula_vector[header.index('OLD20')] = 1
formula_vector[header.index('log10_freq_opensubs')] = -2
formula_vector[header.index('log10_freq_sdewac')] = -2
formula_vector[header.index('concreteness')] = -1

easiness_scores = dict()
for k, v in stimuli.items():
    easiness_scores[k] = -numpy.sum(v*formula_vector)

sorted_ease = sorted(easiness_scores.items(), key=lambda item : item[1], reverse=True)

conc_scores = {k : v[header.index('concreteness')] for k, v in stimuli.items()}

w2v_vectors = dict()
for w in easiness_scores.keys():
    w2v_vectors[w] = w2v.wv[w]

### computing avg sim per word
avg_sims = {k_one : numpy.average([scipy.spatial.distance.cosine(v_one, v_two) for k_two, v_two in w2v_vectors.items()]) for k_one, v_one in w2v_vectors.items()}

matches = dict()
mismatches = dict()

for k_one, v_one in tqdm(w2v_vectors.items()):
    easiness_w = dict()
    for k_two, v_two in w2v_vectors.items():
        if k_one == k_two:
            continue
        w2v_sim = scipy.spatial.distance.cosine(v_one, v_two) / (avg_sims[k_one]*avg_sims[k_two])
        conc_sim = abs(conc_scores[k_one]-conc_scores[k_two])
        easiness_w[k_two] = w2v_sim - 0.5*conc_sim + 0.25*easiness_scores[k_one] + 0.25*easiness_scores[k_two]
    sorted_comb_ease_w = sorted(easiness_w.items(), key=lambda item : item[1])
    match_index = random.choice(range(-10, 0))
    mismatch_index = random.choice(range(0, 10))
    matches[tuple(sorted([k_one, sorted_comb_ease_w[match_index][0]]))] = sorted_comb_ease_w[match_index][1]
    mismatches[tuple(sorted([k_one, sorted_comb_ease_w[mismatch_index][0]]))] = sorted_comb_ease_w[mismatch_index][1]

matches = {k : v for k, v in sorted(matches.items(), key=lambda item : item[1], reverse=True)}
mismatches = {k : v for k, v in sorted(mismatches.items(), key=lambda item : item[1], reverse=True)}

with open('automatic_matches.tsv', 'w') as o:
    o.write('word_one\teasiness_one\tword_two\teasiness_two\trelatedness_judgment_easiness\n')
    for k, v in matches.items():
        o.write('{}\t{}\t{}\t{}\t{}\n'.format(k[0], easiness_scores[k[0]], k[1], easiness_scores[k[1]], v))
with open('automatic_mismatches.tsv', 'w') as o:
    o.write('word_one\teasiness_one\tword_two\teasiness_two\trelatedness_judgment_easiness\n')
    for k, v in mismatches.items():
        o.write('{}\t{}\t{}\t{}\t{}\n'.format(k[0], easiness_scores[k[0]], k[1], easiness_scores[k[1]], v))
