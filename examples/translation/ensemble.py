from itertools import combinations
from glob import glob
import sys
from tqdm import tqdm
import random

tot = sys.argv[1:-2]
filenames = []
for fns in tot:
    filenames += glob(fns)
print(filenames)
r = int(sys.argv[-2])
if r <= 1:
    metafns = [filenames]
else:
    metafns = list(combinations(filenames, r))
random.shuffle(metafns)
samples = int(sys.argv[-1])
metafns = metafns[:samples]

for fns in tqdm(metafns):
    all_scores, labels, all_preds = [], [], []
    print(fns)
    for fname in fns:
        scores, preds, = [], []
        with open(fname, 'r') as fin:
            for line in fin:
                if line.startswith('D'):
                    _, score, decoded = line.split('\t')
                    preds.append(decoded)
                    scores.append(score)
                elif line.startswith('T'):
                    if fname == fns[0] and len(all_scores) == 0:
                        _, label = line.split('\t')
                        labels.append(label)
        assert len(scores) == len(labels) == len(preds)
        all_scores.append(scores)
        all_preds.append(preds)

    best_preds = []
    for i in range(len(labels)):
        best_score = -1
        for j in range(len(all_scores)):
            scores, preds = all_scores[j], all_preds[j]
            scores = [float(s) for s in scores]
            if scores[i] > best_score:
                best_score = scores[i]
                best_pred = preds[i]
        best_preds.append(best_pred)

    outname = '_'.join([x.split('/')[1] for x in fns])
    with open(f"preds/{r}/{outname}.txt", 'w') as fout:
        for text in best_preds:
            fout.write(text)

with open("reference.txt", 'w') as fout:
    for text in labels:
        fout.write(text)
