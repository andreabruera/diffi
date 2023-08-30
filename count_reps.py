import os

file_path = os.path.join('data', 'matches.txt')
f = dict()
with open(file_path) as i:
    counter = 0
    for l in i:
        if counter == 0:
            counter += 1
            continue
        line = l.strip().split('\t')
        if line[1] not in f.keys():
            f[line[1]] = 1
        else:
            f[line[1]] += 1
        if line[2] not in f.keys():
            f[line[2]] = 1
        else:
            f[line[2]] += 1
print(sorted(f.items(), key=lambda item : item[1], reverse=True)[0])
