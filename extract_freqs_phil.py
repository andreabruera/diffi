import numpy
import os
import pickle

### reading phil's candidate dataset
relevant_words = list()
with open(os.path.join('data', 'german_nouns_phil.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            continue
        relevant_words.append(line[0])

with open(os.path.join('pickles', 'sdewac_lemma_freqs.pkl'), 'rb') as i:
    lemma_freqs = pickle.load(i)
with open(os.path.join('pickles', 'sdewac_word_freqs.pkl'), 'rb') as i:
    word_freqs = pickle.load(i)

with open(os.path.join('data', 'german_nouns_phil_freqs.tsv'), 'w') as o:
    o.write('word\traw_word_frequency_sdewac\tlog10_word_frequency_sdewac\traw_lemma_frequency_sdewac\tlog10_lemma_frequency_sdewac\n')
    for w in relevant_words:
        o.write('{}\t'.format(w))
        try:
            o.write('{}\t'.format(word_freqs[w]))
            o.write('{}\t'.format(numpy.log10(word_freqs[w])))
        except KeyError:
            o.write('na\tna\t')
        o.write('{}\t'.format(lemma_freqs[w]))
        o.write('{}\t'.format(numpy.log10(lemma_freqs[w])))
        o.write('\n')
