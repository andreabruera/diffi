import os

### reading words
words = dict()
with open(os.path.join('data', 'german_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            words_header = line.copy()
        words[line[0]] = line[1:]
norms = dict()
with open(os.path.join('data', 'nouns_phil_semantic_norms.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            norms_header = line.copy()
        norms[line[0]] = line[1:]
proto = dict()
with open(os.path.join('data', 'prototypicality_german_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            proto_header = line.copy()
        proto[line[0]] = line[1:]
freqs = dict()
with open(os.path.join('data', 'german_nouns_phil_freqs.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            freqs_header = line.copy()
        freqs[line[0]] = line[1:]
olds = dict()
with open(os.path.join('data', 'old20_scores_candidate_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            olds_header = line.copy()
        olds[line[0]] = line[1:]

import pdb; pdb.set_trace()
all_together = {k : [v] + [len(v)] +freqs[k] + olds[k] + norms[k] + proto[k] for k, v in words.items()}
with open('german_nouns_automated.tsv', 'w') as o:
    o.write('\t'.join(words_header+['word_length']+freqs_header+olds_header+norms_header+proto_header))
    o.write('\n')
    for k, v in all_together.items():
        o.write('{}\t'.format(k))
        o.write('\t'.join(v))
        o.write('\n')
