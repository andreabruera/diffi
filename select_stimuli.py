import gensim
import nltk
import numpy
import os
import random
import scipy
import sklearn

from gensim.models import Word2Vec
from scipy import spatial
from sklearn import metrics
from tqdm import tqdm
from nltk.corpus import wordnet

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

def cosine_similarity(vec_one, vec_two):
    num = sum([a*b for a, b in zip(vec_one, vec_two)])
    den_one = sum([a**2 for a in vec_one])**.5
    den_two = sum([a**2 for a in vec_two])**.5
    cos = num / (den_one * den_two)

    return cos

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
        #if 'nan' not in line:
        if 'nan' not in line and line[0] in w2v.wv.key_to_index.keys():
            if line[0].capitalize() == line[0]:
                stimuli[line[0]] = numpy.array(line[1:], dtype=numpy.float32)
                assert stimuli[line[0]].shape == (len(header), )

'''
import googletrans

from googletrans import Translator, constants

translations = {k : '' for k in stimuli.keys()}
translator = Translator(service_urls=['translate.googleapis.com'])

for k in stimuli.keys():
    translation = translator.translate(k, src='de', dest="en").text
    translations[k] = translation.lower()

with open('stimuli_translations.tsv', 'w') as o:
    for k, v in translations.items():
        o.write('{}\t{}\n'.format(k, v))
'''


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

### z-scoring
mean_easiness = numpy.average(list(easiness_scores.values()))
std_easiness = numpy.std(list(easiness_scores.values()))
easiness_scores = {k : (v-mean_easiness) / std_easiness for k, v in easiness_scores.items()}

conc_scores = {k : v[header.index('concreteness')] for k, v in stimuli.items()}

w2v_similarities_file = os.path.join('models', 'w2v_similarities.tsv')
if os.path.exists(w2v_similarities_file):
    w2v_sims = dict()
    with open(w2v_similarities_file) as i:
        counter = 0
        for l in i:
            if counter == 0:
                counter += 1
                continue
            line = l.strip().split('\t')
            assert len(line) == 3
            w2v_sims[(line[0], line[1])] = float(line[2])
else:
    w2v_vectors = dict()
    for w in easiness_scores.keys():
        w2v_vectors[w] = w2v.wv[w]
    w2v_sims = dict()
    for k_one, v_one in tqdm(w2v_vectors.items()):
        for k_two, v_two in w2v_vectors.items():
            if k_one == k_two:
                continue
            key = tuple(sorted([k_one, k_two]))
            if key in w2v_sims.keys():
                continue
            w2v_sim = sklearn.metrics.pairwise.cosine_similarity(v_one.reshape(1, -1), v_two.reshape(1, -1))[0][0]
            w2v_sims[key] = w2v_sim
    with open(w2v_similarities_file, 'w') as o:
        o.write('word_one\tword_two\tw2v_similarity\n')
        for k, v in w2v_sims.items():
            o.write('{}\t{}\t{}\n'.format(k[0], k[1], float(v)))

lev_sims = {k : levenshtein(k[0], k[1]) for k in w2v_sims.keys()}
### z-scoring
mean_lev = numpy.average(list(lev_sims.values()))
std_lev = numpy.std(list(lev_sims.values()))
lev_sims = {k : (v-mean_lev) / std_lev for k, v in lev_sims.items()}

matches = dict()
mismatches = dict()

mismatch_counter = {w : 0 for w in stimuli.keys()}

merged_similarities = dict()

for k, v in w2v_sims.items():
    k_one = k[0]
    k_two = k[1]
    conc_sim = abs(conc_scores[k_one]-conc_scores[k_two])
    sim_score = v - conc_sim
    merged_similarities[k] = sim_score

### for each word, selecting 2 best / worst similarities

for k_one in stimuli.keys():
    easiness_w = dict()
    for k_two in stimuli.keys():
        if k_one == k_two:
            continue

        easiness_w[k_two] = merged_similarities[tuple(sorted([k_one, k_two]))]
    ### matches
    sorted_comb_ease_w = sorted(easiness_w.items(), key=lambda item : item[1])
    match_indices = random.sample(range(-4, 0), k=2)
    ### mismatches
    sorted_comb_ease_w = sorted([(k, v) for k, v in easiness_w.items() if mismatch_counter[k]<3], key=lambda item : item[1])
    mismatch_indices = random.sample(range(0, 4), k=2)
    for match_index, mismatch_index in zip(match_indices, mismatch_indices):
        match_word = sorted_comb_ease_w[match_index][0]
        matches[tuple(sorted([k_one, match_word]))] = sorted_comb_ease_w[match_index][1]
        mismatch_word = sorted_comb_ease_w[mismatch_index][0]
        mismatch_counter[mismatch_word] += 1
        mismatches[tuple(sorted([k_one, mismatch_word]))] = sorted_comb_ease_w[mismatch_index][1]

### matches - maximum number of repetitions: 5 (replication of the stimuli by Wilson et al. 2018)
matches_counter = {w : 0 for w in stimuli.keys()}
for w_one, w_two in matches.keys():
    matches_counter[w_one] += 1
    matches_counter[w_two] += 1
for w, count in matches_counter.items():
    if count > 5:
        check_counter = {w : 0 for w in stimuli.keys()}
        for w_one, w_two in matches.keys():
            check_counter[w_one] += 1
            check_counter[w_two] += 1
        if check_counter[w] < 5:
            continue
        to_be_checked = [(k, v) for k, v in matches.items() if w in k]
        to_be_removed = sorted(to_be_checked, key = lambda item : item[1])[:-5]
        assert len(to_be_checked) - len(to_be_removed) == 5
        for k in to_be_removed:
            del matches[k[0]]
matches_counter = {w : 0 for w in stimuli.keys()}
for w_one, w_two in matches.keys():
    matches_counter[w_one] += 1
    matches_counter[w_two] += 1
assert max(list(matches_counter.values())) == 5

### mismatches - maximum number of repetitions: 5
mismatches_counter = {w : 0 for w in stimuli.keys()}
for w_one, w_two in mismatches.keys():
    mismatches_counter[w_one] += 1
    mismatches_counter[w_two] += 1
assert max(list(mismatches_counter.values())) == 5

matches = {k : v for k,v in sorted(matches.items(), key=lambda item : item[1], reverse=True)[:2700]}
mismatches = {k : v for k,v in sorted(mismatches.items(), key=lambda item : item[1])[:2700]}

### z-scoring
mean_sims = numpy.average(list(matches.values())+list(mismatches.values()))
std_sims = numpy.std(list(matches.values())+list(mismatches.values()))
matches = {k : (v-mean_sims) / std_sims for k, v in matches.items()}
mismatches = {k : (v-mean_sims) / std_sims for k, v in mismatches.items()}

### matches are sorted from higher to lower
final_matches = {k : (v, v+0.25*easiness_scores[k[0]]+.25*easiness_scores[k[1]]-lev_sims[k]) for k, v in matches.items()}
final_matches = {k : v for k, v in sorted(final_matches.items(), key=lambda item : item[1][1], reverse=True)}
### mismatches are sorted from lower to higher
final_mismatches = {k : (v, -v+.25*easiness_scores[k[0]]+.25*easiness_scores[k[1]]-lev_sims[k]) for k, v in mismatches.items()}
final_mismatches = {k : v for k, v in sorted(final_mismatches.items(), key=lambda item : item[1][1], reverse=True)}

output_folder = 'selected_stimuli'
os.makedirs(output_folder, exist_ok=True)

backup_matches = list()
backup_mismatches = list()

with open(os.path.join(output_folder, 'automatic_matches.tsv'), 'w') as o:
    intro = 'word_one\teasiness_word_one\tword_two\teasiness_word_two\teasiness_both_words\tdifficulty_category\n'
    o.write(intro)
    backup_matches.append(intro)
    category_counter = 1
    line_counter = 0
    backup_counter = 0
    for k, v in final_matches.items():
        line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(k[0], easiness_scores[k[0]], k[1], easiness_scores[k[1]], v[1], category_counter)
        if line_counter <= 300:
            line_counter += 1
            o.write(line)
        elif backup_counter < 100:
            backup_counter += 1
            backup_matches.append(line)
        else:
            category_counter += 1
            backup_counter = 0
            line_counter = 1
            o.write(line)
with open(os.path.join(output_folder, 'automatic_mismatches.tsv'), 'w') as o:
    intro = 'word_one\teasiness_word_one\tword_two\teasiness_word_two\teasiness_both_words\tdifficulty_category\n'
    o.write(intro)
    backup_mismatches.append(intro)
    category_counter = 1
    line_counter = 0
    backup_counter = 0
    for k, v in final_mismatches.items():
        line = '{}\t{}\t{}\t{}\t{}\t{}\n'.format(k[0], easiness_scores[k[0]], k[1], easiness_scores[k[1]], v[1], category_counter)
        if line_counter <= 300:
            line_counter += 1
            o.write(line)
        elif backup_counter < 100:
            backup_counter += 1
            backup_mismatches.append(line)
        else:
            category_counter += 1
            backup_counter = 0
            line_counter = 1
            o.write(line)
with open(os.path.join(output_folder, 'backup_automatic_matches.tsv'), 'w') as o:
    for l in backup_matches:
        o.write(l)
with open(os.path.join(output_folder, 'backup_automatic_mismatches.tsv'), 'w') as o:
    for l in backup_mismatches:
        o.write(l)
