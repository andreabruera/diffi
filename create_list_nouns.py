import os

with open(os.path.join('data', 'phil_predicted_aligned_semantic_norms.tsv')) as i:
    with open(os.path.join('data', 'german_nouns_phil.tsv'), 'w') as o:
        for l in i:
            line = l.strip().split('\t')[:2]
            o.write('\t'.join(line))
            o.write('\n')
