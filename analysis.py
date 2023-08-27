import gensim
import itertools
import matplotlib
import nltk
import numpy
import os
import re
import sklearn

from gensim import downloader
from matplotlib import pyplot
from nltk.corpus import wordnet

### read the very bizarre MRC dataset

rel_idxs = {
            'mrc_concreteness' : (28, 31),
            'mrc_AoA' : (40, 43),
            }

norms = {k : dict() for k in rel_idxs.keys()}
to_be_corrected = {
                   'harbour' : 'harbor',
                   'neighbourhood' : 'neighborhood',
                   'vapour' : 'vapor',
                   'rumour' : 'rumor',
                   'doughnut' : 'donut',
                   'neighbour' : 'neighbor',
                   'humour' : 'humor',
                   'demeanour' : 'demeanor',
                   'flavour' : 'flavor',
                   'pretence' : 'pretense',
                   'odour' : 'odor',
                   'discolouration' : 'discoloration',
                   'axe' : 'ax',
                   'judgement' : 'judgment',
                   }

with open(os.path.join('norms', 'mrc2.dct')) as i:
    for l in i:
        word = l[51:].split('|')[0].lower()
        if word in to_be_corrected.keys():
            word = to_be_corrected[word]
        for var, idxs in rel_idxs.items():
            val = int(l[idxs[0]:idxs[1]])
            if val != 0:
                norms[var][word] = val

### all
all_values = {
              'matches' : dict(),
              'mismatches' : dict(),
              }

### read norms

folder = 'norms'
assert len(os.listdir(folder)) == 5
associations = dict()
pos = dict()

for f in os.listdir(folder):
    counter = 0
    if 'Concreteness' in f:
        keys = ['concreteness']
        all_rel_idxs = [[0, 2]]
    elif 'AoA' in f:
        keys = ['AoA']
        all_rel_idxs = [[0, 3]]
    elif 'PoS' in f:
        keys = ['log10_contextual_diversity', 'log10_freq']
        all_rel_idxs = [[0, 8], [0, 6]]
    elif 'strength' in f:
        keys = ['associative_strength']
        all_rel_idxs = [[0, 1, -1]]
    else:
        continue

    for key, rel_idxs in zip(keys, all_rel_idxs):

        norms[key] = dict()
        with open(os.path.join(folder, f), errors='ignore') as i:
            for l in i:
                if counter == 0:
                    counter += 1
                    continue
                line = l.strip().split('\t')
                if len(line) < 3:
                    continue
                if key == 'associative_strength':
                    associations[(line[0], line[1])] = float(line[-1])
                else:
                    try:
                        norms[key][line[rel_idxs[0]]] = float(line[rel_idxs[1]])
                    except ValueError:
                        continue
                    if key == 'concreteness':
                        pos[line[0]] = line[-1]
del norms['associative_strength']

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
missing_words = [w for w in words_total if w not in norms['mrc_AoA']]
words_per_cat = {k : set([w for case in lst for w in case[:2]]) for k, lst in couples_dict.items()}

### loading w2v
wv = gensim.downloader.load('word2vec-google-news-300')

### printouts!
r_dataset = {k : dict() for k in norms.keys()}
r_dataset['w2v_distance'] = dict()
r_dataset['wordnet_distance'] = dict()
r_dataset['difference_wordnet_senses'] = dict()

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
                r_dataset[key][tuple(tup[:2])] = 'na'
                continue
            diff = abs(current_variable[tup[0]]-current_variable[tup[1]])
            r_dataset[key][tuple(tup[:2])] = diff
            values.append(diff)
        print('avg difference in {} for {}: {}'.format(key, k, numpy.average(values)))
        print('std of differences in {} for {}: {}'.format(key, k, numpy.std(values)))
        all_values[k]['difference_ in {}'.format(key)] = values
    ### similarities
    ### w2v
    print('\n')
    w2v_sims = [wv.similarity(tup[0], tup[1]) for tup in tuples]
    for t, s in zip(tuples, w2v_sims):
        r_dataset['w2v_distance'][(t[0], t[1])] = 1-s
    print('avg w2v similarity for {}: {}'.format(k, numpy.average(w2v_sims)))
    all_values[k]['word2vec_distance'] = [1-s for s in w2v_sims]
    print('std of w2v similarity for {}: {}'.format(k, numpy.std(w2v_sims)))
    ### associations
    #r_dataset['word_association_distance'] = dict()
    assos = list()
    missing_tuples = list()
    for tup in tuples:
        t = tuple(tup[:2])
        if t not in associations.keys():
            missing_tuples.append(t)
            #r_dataset['word_association_distance'][t] = 'na'
        else:
            asso = associations[t]
            #r_dataset['word_association_distance'][t] = 1-asso
            assos.append(asso)
    print('missing tuples for word associations: {}'.format(len(missing_tuples)))
    print('avg word association for {}: {}'.format(k, numpy.average(assos)))
    all_values[k]['word_association_distance'] = [1-s for s in assos]
    print('std of word association for {}: {}'.format(k, numpy.std(assos)))

    ### wordnet
    wn_sims = list()
    diff_senses = list()
    print('\n')
    counter = 0
    for tup in tuples:
        counter += 1
        t = tuple(tup[:2])
        #if tup[0] in missing_words or tup[1] in missing_words:
        #    continue
        words = list()
        tup_sims = list()
        for w in t:
            #current_pos = pos[w]
            w_syn = wordnet.synsets(w)
            words.append(w_syn)
        senses = abs(len(words[0])-len(words[1]))
        r_dataset['difference_wordnet_senses'][t] = senses
        diff_senses.append(senses)
        combs = list(itertools.product(words[0], words[1]))
        for c in combs:
            #sim = wordnet.path_similarity(c[0], c[1])
            sim = wordnet.wup_similarity(c[0], c[1])
            tup_sims.append(sim)
        avg_sims = numpy.average(tup_sims)
        wn_sims.append(avg_sims)
        r_dataset['wordnet_distance'][t] = 1-avg_sims
    print(counter)
    all_values[k]['wordnet_distance'] = [1-s for s in wn_sims]
    all_values[k]['difference_in number_of wordnet_senses'] = diff_senses
    print('avg wordnet similarity for {}: {}'.format(k, numpy.average(wn_sims)))
    print('std of wordnet similarity for {}: {}'.format(k, numpy.std(wn_sims)))
    print('avg differences in senses for {}: {}'.format(k, numpy.average(diff_senses)))
    print('std of differences in senses for {}: {}'.format(k, numpy.std(diff_senses)))

### standardizing
keys = list(all_values[k].keys())
parameters_est = {k : [val for v in all_values.values() for val_key, vals in v.items() for val in vals if val_key==k] for k in keys}
parameters = {k : [numpy.average(v), numpy.std(v)] for k, v in parameters_est.items()}
z_scores = {k : {k_two : list() for k_two in v.keys()} for k, v in all_values.items()}
for k, v in all_values.items():
    for k_two, v_two in v.items():
        vals = [(score - parameters[k_two][0])/parameters[k_two][1] for score in v_two]
        z_scores[k][k_two] = vals

### now plotting
plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)
out_file = os.path.join(plot_folder, 'analysis_stimuli_wilson.jpg')
fig, ax = pyplot.subplots(figsize=(22, 10), constrained_layout=True)

matched = [(var, var_data) for var, var_data in z_scores['matches'].items()]
mismatched = [(var, var_data) for var, var_data in z_scores['mismatches'].items()]
assert [v[0] for v in matched] == [v[0] for v in mismatched]
xs = [v[0] for v in matched]
matched = [v[1] for v in matched]
mismatched = [v[1] for v in mismatched]
v1 = ax.violinplot(matched,
                       #points=100,
                       positions=range(len(xs)),
                       showmeans=True,
                       showextrema=False,
                       showmedians=False,
                       )
for b in v1['bodies']:
    # get the center
    m = numpy.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], -numpy.inf, m)
    b.set_color('darkorange')
v1['cmeans'].set_color('darkorange')
v2 = ax.violinplot(mismatched,
                       #points=100,
                       positions=range(len(xs)),
                       showmeans=True,
                       showextrema=False,
                       showmedians=False,
                       )
for b in v2['bodies']:
    # get the center
    m = numpy.mean(b.get_paths()[0].vertices[:, 0])
    # modify the paths to not go further right than the center
    b.get_paths()[0].vertices[:, 0] = numpy.clip(b.get_paths()[0].vertices[:, 0], m, numpy.inf)
    b.set_color('teal')
v2['cmeans'].set_color('teal')
ax.legend(
          [v1['bodies'][0], v2['bodies'][0]],
          ['matches', 'mismatches'],
          fontsize=20
          )
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(
                    [x.replace('_', '\n') for x in xs],
                    fontsize=23,
                    fontweight='bold',
                    )
pyplot.yticks(fontsize=15)
ax.set_ylabel(
              'Standardized value',
              fontsize=20,
              fontweight='bold',
              labelpad=20
              )
ax.set_title(
             'Values',
             pad=20,
             fontweight='bold',
             fontsize=25,
             )
pyplot.savefig(out_file)

### preparing r dataset
with open('r_data.tsv', 'w') as o:
    o.write('label\taoa\tconcreteness_difference\tlog10_frequency\tcd_distance\tw2v_distance\twordnet_distance\tsenses\n')
    for label, tuples in couples_dict.items():
        if label == 'matches':
            label = +1
        else:
            label = -1
        for tup in tuples:
            #print(tup)
            t = tuple(tup[:2])
            conc = r_dataset['mrc_concreteness'][t]
            #conc = r_dataset['concreteness'][t]
            aoa = r_dataset['mrc_AoA'][t]
            w2v = r_dataset['w2v_distance'][t]
            wordnet = r_dataset['wordnet_distance'][t]
            senses = r_dataset['difference_wordnet_senses'][t]
            cd = r_dataset['log10_contextual_diversity'][t]
            freq = r_dataset['log10_freq'][t]
            line = [label, aoa, conc, freq, cd, w2v, wordnet, senses]
            #print(line)
            #assert 'na' not in line
            if 'na' not in line:
                #print(line)
                o.write('\t'.join([str(v) for v in line]))
                o.write('\n')
