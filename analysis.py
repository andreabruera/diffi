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

### all
all_values = {
              'matches' : dict(), 
              'mismatches' : dict(),
              }

### read norms

folder = 'norms'
assert len(os.listdir(folder)) == 3
norms = dict()
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
        all_values[k]['difference_ in {}'.format(key)] = values
    ### similarities
    ### w2v
    print('\n')
    w2v_sims = [wv.similarity(tup[0], tup[1]) for tup in tuples]
    print('avg w2v similarity for {}: {}'.format(k, numpy.average(w2v_sims)))
    all_values[k]['word2vec_distance'] = [1-s for s in w2v_sims]
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
