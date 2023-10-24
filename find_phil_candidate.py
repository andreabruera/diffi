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

'''
### frequency threshold: 1000
freq_threshold = 1000
nouns_candidates = list()
for w, freq in lemma_freqs.items():
    if freq > freq_threshold:
        w_pos = sorted(lemma_pos[w].items(), key=lambda item : item[1], reverse=True)
        if w_pos[0][0] == 'NN':
            if w.lower() in ft_de.keys():
                nouns_candidates.append(w)
'''
nouns_candidates = list()
trans_de = dict()
with open(os.path.join('data', 'german_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            continue
        nouns_candidates.append(line[0])
        trans_de[line[0]] = line[1]

### loading aligned english fasttext
ft_en_file = os.path.join('pickles', 'ft_en_aligned.pkl')
if os.path.exists(ft_en_file):
    with open(ft_en_file, 'rb') as i:
        ft_en = pickle.load(i)
else:
    ft_en = dict()
    with open(os.path.join('..', '..', 'dataset', 'word_vectors', 'en', 'wiki.en.align.vec')) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split(' ')
            if l_i == 0:
                continue
            ft_en[line[0]] = numpy.array(line[1:], dtype=numpy.float64)
    with open(ft_en_file, 'wb') as i:
        pickle.dump(ft_en, i)

'''
### learning a transformation of tf_de words
translator = googletrans.Translator()
trans_de = dict()
for w in tqdm(nouns_candidates):
    trans = translator.translate(w, src='de', dest='en').text.lower()
    trans_de[w] = trans

### checking alignment
sims = dict()
for w, vec in ft_en.items():
    try:
        sim = 1 - scipy.spatial.distance.cosine(vec, ft_de['brot'])
    except ValueError:
        continue
    sims[w] = sim
top_twenty = sorted(sims.items(), key=lambda item: item[1], reverse=True)[:20]
'''

perceptual_norms = {
        ### concreteness
         'concreteness' : dict(),
        ### senses
         'visual' : dict(),
         'haptic' : dict(),
         'gustatory' : dict(),
         'olfactory' : dict(),
         'auditory' : dict(),
        ### body
         'hand' : dict(),
         'leg' : dict(),
         'head' : dict(),
         'mouth' : dict(),
         'torso' : dict(),
         }

### loading concreteness
with open(os.path.join('norms', 'Concreteness_ratings_Brysbaert_et_al_BRM.txt')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        if line[header.index('Word')] not in ft_en.keys():
            continue
        perceptual_norms['concreteness'][line[header.index('Word')]] = float(line[header.index('Conc.M')])

### loading sensorimotor
with open(os.path.join('norms', 'Lancaster_sensorimotor_norms_for_39707_words.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        word = line[header.index('Word')].lower()
        if word not in perceptual_norms['concreteness'].keys():
            continue
        perceptual_norms['visual'][word] = float(line[header.index('Visual.mean')])
        perceptual_norms['haptic'][word] = float(line[header.index('Haptic.mean')])
        perceptual_norms['gustatory'][word] = float(line[header.index('Gustatory.mean')])
        perceptual_norms['olfactory'][word] = float(line[header.index('Olfactory.mean')])
        perceptual_norms['auditory'][word] = float(line[header.index('Auditory.mean')])
        perceptual_norms['hand'][word] = float(line[header.index('Hand_arm.mean')])
        perceptual_norms['leg'][word] = float(line[header.index('Foot_leg.mean')])
        perceptual_norms['head'][word] = float(line[header.index('Head.mean')])
        perceptual_norms['mouth'][word] = float(line[header.index('Mouth.mean')])
        perceptual_norms['torso'][word] = float(line[header.index('Torso.mean')])
### correcting for a very annoying mistake on the dataset
perceptual_norms = {k : {w : val if val<=5 else float('.{}'.format(str(int(val)))) for w, val in v.items()} for k,v in perceptual_norms.items()}

perceptual_keys = list(perceptual_norms.keys()).copy()
### loading ridge model
#ridge_model = sklearn.linear_model.RidgeCV(alphas=(0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000))
perceptual_predictions = dict()
for k in perceptual_keys:
    print(k)
    ridge_model = sklearn.linear_model.Ridge()
    training_input = [ft_en[w] for w in perceptual_norms['visual'].keys()]
    training_target = [[perceptual_norms[k][w]] for w in perceptual_norms['visual'].keys()]
    ridge_model.fit(training_input, training_target)
    raw_perceptual_predictions = {w : p[0] for w, p in zip(nouns_candidates, ridge_model.predict([ft_de[w.lower()] for w in nouns_candidates]))}

    perc_means = numpy.mean([v for v in raw_perceptual_predictions.values()])
    #assert len(perc_means) == len(perceptual_keys)
    perc_stds = numpy.std([v for v in raw_perceptual_predictions.values()])
    #assert len(perc_stds) == len(perceptual_keys)
    ### scaling 0-5 to match en
    #vals = [raw_perceptual_predictions[w] for w in nouns_candidates]
    #den = max(vals)-min(vals)
    #scaled_vals = numpy.array([((v - min(vals)) / den)*5 for v in vals])
    #scaled_vals = zero_one_vals * (5 - 0) + 0

    perceptual_predictions[k] = {k : (v-perc_means)/perc_stds for k, v in raw_perceptual_predictions.items()}
    #perceptual_predictions[k] = {k : round(v, 3) for k, v in zip(nouns_candidates, scaled_vals)}

emotional_norms = {
        ### emotion
         'valence' : dict(),
         'arousal' : dict(),
         'dominance' : dict(),
         }
### loading emotions
with open(os.path.join('norms', 'BRM-emot-submit.csv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split(',')
        if l_i == 0:
            header = line.copy()
            continue
        if line[header.index('Word')] not in ft_en.keys():
            continue
        emotional_norms['valence'][line[header.index('Word')]] = float(line[header.index('V.Mean.Sum')])
        emotional_norms['arousal'][line[header.index('Word')]] = float(line[header.index('A.Mean.Sum')])
        emotional_norms['dominance'][line[header.index('Word')]] = float(line[header.index('D.Mean.Sum')])

emotional_keys = list(emotional_norms.keys()).copy()
emotional_predictions = dict()
for k in emotional_keys:
    ridge_model = sklearn.linear_model.Ridge()
    training_input = [ft_en[w] for w in emotional_norms['valence'].keys()]
    training_target = [[emotional_norms[k][w]] for w in emotional_norms['valence'].keys()]
    ridge_model.fit(training_input, training_target)
    raw_emotional_predictions = {w : p[0] for w, p in zip(nouns_candidates, ridge_model.predict([ft_de[w.lower()] for w in nouns_candidates]))}

    perc_means = numpy.mean([v for v in raw_emotional_predictions.values()])
    ##assert len(perc_means) == len(perceptual_keys)
    perc_stds = numpy.std([v for v in raw_emotional_predictions.values()])
    #assert len(perc_stds) == len(perceptual_keys)
    #vals = [raw_emotional_predictions[w] for w in nouns_candidates]
    #den = max(vals)-min(vals)
    #scaled_vals = numpy.array([((v - min(vals)) / den)*5 for v in vals])
    #scaled_vals = zero_one_vals * (5 - 0) + 0

    emotional_predictions[k] = {k : (v-perc_means)/perc_stds for k, v in raw_emotional_predictions.items()}
    #emotional_predictions[k] = {k : round(v, 3) for k, v in zip(nouns_candidates, scaled_vals)}


with open(os.path.join('data', 'nouns_phil_semantic_norms.tsv'), 'w') as o:
    ### header
    o.write('word\t')
    o.write('en_google_translation\t')
    ### german
    for k in perceptual_keys:
        o.write('predicted_{}\t'.format(k))
    for k in emotional_keys:
        o.write('predicted_{}\t'.format(k))
    ### english
    for k in perceptual_keys:
        o.write('en_{}\t'.format(k))
    for k in emotional_keys:
        o.write('en_{}\t'.format(k))
    ### newline
    o.write('\n')
    ###
    ### writing values
    for w in nouns_candidates:
        o.write('{}\t'.format(w))
        o.write('{}\t'.format(trans_de[w]))
        ### predicted
        for k_i, k in enumerate(perceptual_keys):
            o.write('{}\t'.format(round(perceptual_predictions[k][w], 3)))
        for k_i, k in enumerate(emotional_keys):
            o.write('{}\t'.format(emotional_predictions[k][w]))
        ### english
        for k_i, k in enumerate(perceptual_keys):
            try:
                norm = perceptual_norms[k][trans_de[w]]
            except KeyError:
                norm = 'na'
            o.write('{}\t'.format(norm))
        for k_i, k in enumerate(emotional_keys):
            try:
                norm = emotional_norms[k][trans_de[w]]
            except KeyError:
                norm = 'na'
            o.write('{}\t'.format(norm))
        o.write('\n')
