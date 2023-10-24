import numpy
import os
import scipy

from scipy import stats

data = {
        'en' : dict(),
        'predicted' : dict(),
        }
with open(os.path.join('data', 'nouns_phil_semantic_norms.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        if 'na' in line:
            continue
        for h_i, h in enumerate(header):
            if 'google' in h:
                continue
            if 'en_' in h or 'predicted_' in h:
                typ = h.split('_')[0]
                split_h = h.split('_')[1]
                val = float(line[h_i])
                try:
                    data[typ][split_h].append(val)
                except KeyError:
                    data[typ][split_h] = [val]

for k, en_data in data['en'].items():
    pred_data = data['predicted'][k]
    corr = scipy.stats.pearsonr(en_data, pred_data)
    print([k, corr])
