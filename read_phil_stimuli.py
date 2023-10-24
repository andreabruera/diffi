import os

folder = 'phil'
phil_data = dict()

for f in os.listdir(folder):
    if 'tsv' not in f:
        continue
    dataset = f.replace('.tsv', '')
    phil_data[dataset] = dict()
    with open(os.path.join(folder, f)) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = line.copy()
                if '9' in dataset:
                    header = header[:-6]
                for h in header:
                    phil_data[dataset][h] = list()
                continue
            if '46' in dataset:
                if len(line) != len(header):
                    print('dataset {}, error with: {}'.format(dataset, line[header.index('Words')]))
                    continue
            if '9' in dataset:
                if len(line) < len(header):
                    print('dataset {}, error with: {}'.format(dataset, line[header.index('Words')]))
                    continue
            for h_i, h in enumerate(header):
                phil_data[dataset][h].append(line[h_i])

### now bringing it all together
relevant_keys = set(phil_data['v46'].keys()).intersection(set(phil_data['v9'].keys()))

all_phil = {k : list() for k in relevant_keys if k!='Included'}
for _, d in phil_data.items():
    total = len(d['Words'])
    for i in range(total):
        for h in relevant_keys:
            if h == 'Included':
                continue
            val = d[h][i].replace(',', '.')
            try:
                val = float(val)
            except ValueError:
                pass
            all_phil[h].append(val)

### checking duplicates
counter = {w : 0 for w in all_phil['Words']}
for w in all_phil['Words']:
    counter[w] += 1

to_be_checked = [w for w,c in counter.items() if c>1]
to_be_removed = list()
idxs = {w : [idx for idx, idx_w in enumerate(all_phil['Words']) if idx_w==w] for w in to_be_checked}
for w, w_idxs in idxs.items():
    assert len(w_idxs) == 2
    one = [l[w_idxs[0]] for l in all_phil.values()]
    two = [l[w_idxs[1]] for l in all_phil.values()]
    try:
        assert one == two
        to_be_removed.append(w_idxs[1])
    except AssertionError:
        #print(w)
        #print(one)
        #print(two)
        to_be_removed.append(w_idxs[0])
        to_be_removed.append(w_idxs[1])

final_phil = {k : [v for v_i, v in enumerate(d) if v_i not in to_be_removed] for k, d in all_phil.items()}
assert len(list(set([len(v) for v in final_phil.values()]))) == 1

clean_phil = dict()
for k, v in final_phil.items():
    if k == 'Words':
        clean_phil[k] = v
    else:
        clean_phil[k] = [val if type(val)==float else 'na' for val in v]

keys = sorted(clean_phil.keys())
total = len(clean_phil['Words'])

with open(os.path.join('data', 'phil_clean.tsv'), 'w') as o:
    o.write('\t'.join(keys))
    o.write('\n')
    for i in range(total):
        for k in keys:
            o.write('{}\t'.format(clean_phil[k][i]))
        o.write('\n')
