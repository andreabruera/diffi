import numpy
import os
import scipy

from scipy import stats

predicted_words = list()
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
        word = line[header.index('word')]
        predicted_words.append(word)
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

print('evaluation against english')
for k, en_data in data['en'].items():
    pred_data = data['predicted'][k]
    corr = scipy.stats.pearsonr(en_data, pred_data)
    print([k, corr])

### reading phil's data
phil_data = {'de' : dict(), 'predicted' : dict()}
with open(os.path.join('data', 'phil_clean.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        word = line[header.index('Words')]
        if word not in predicted_words:
            continue
        try: 
            phil_data['de']['auditory'].append(float(line[header.index('Geraeusch')]))
            phil_data['de']['hand'].append(float(line[header.index('Handlung')]))
        except KeyError:
            phil_data['de']['auditory'] = [float(line[header.index('Geraeusch')])]
            phil_data['de']['hand'] = [float(line[header.index('Handlung')])]

predictions = dict()
with open(os.path.join('data', 'nouns_phil_semantic_norms.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        if 'na' in line:
            continue
        word = line[header.index('word')]
        predictions[word] = [float(line[header.index('predicted_auditory')]), float(line[header.index('predicted_hand')])]

with open(os.path.join('data', 'phil_clean.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            continue
        word = line[header.index('Words')]
        if word not in predicted_words:
            print(word)
            continue
        try: 
            phil_data['predicted']['auditory'].append(predictions[word][0])
            phil_data['predicted']['hand'].append(predictions[word][1])
        except KeyError:
            phil_data['predicted']['auditory'] = [predictions[word][0]]
            phil_data['predicted']['hand'] = [predictions[word][1]]


print('evaluation against german, using {} words'.format(len(phil_data['predicted']['hand'])))
for k, de_data in phil_data['de'].items():
    pred_data = phil_data['predicted'][k]
    corr = scipy.stats.pearsonr(de_data, pred_data)
    print([k, corr])
