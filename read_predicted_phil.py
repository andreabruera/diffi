import numpy
import os

predicted = dict()
with open(os.path.join('data', 'phil_predicted_aligned_semantic_norms.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.replace('[', '').replace(']', '').strip().split('\t')
        if l_i == 0:
            header = line.copy()
            predicted = {h : list() for h in header}
            continue
        for h_i, h in enumerate(header):
            try:
                predicted[h].append(float(line[h_i]))
            except ValueError:
                predicted[h].append(line[h_i])

### sample: gustatory

conc = [predicted['word'][k[0]] for k in sorted(enumerate(predicted['concreteness']), key=lambda item : item[1], reverse=True)]
gust = [predicted['word'][k[0]] for k in sorted(enumerate(predicted['gustatory']), key=lambda item : item[1], reverse=True)]
hand = [predicted['word'][k[0]] for k in sorted(enumerate(predicted['hand']), key=lambda item : item[1], reverse=True)]
auditory = [predicted['word'][k[0]] for k in sorted(enumerate(predicted['auditory']), key=lambda item : item[1], reverse=True)]
import pdb; pdb.set_trace()
