from utils_metrics import metrics_generation
import sys
trues_list = []
preds_list = []

for line in open(sys.argv[1]):
    line = line.strip().split("|")
    trues_list.append(line[0])
    preds_list.append(line[1])

scores = metrics_generation(trues_list, preds_list)

print("bleu score:", scores["bleu"])

from collections import defaultdict

trues = [ [] for _ in range(50)]
preds = [ [] for _ in range(50)]

for t, p in zip(trues_list, preds_list):
    l = len(t.split())
    if l <= 5:
        trues[5].append(t)
        preds[5].append(p)
    elif l >= 25:
        trues[25].append(t)
        preds[25].append(p)
    else:
        trues[l//2*2].append(t)
        preds[l//2*2].append(p)

for i in range(50):
    if len(trues[i]) == 0:
        continue
    print("({},{})".format(i, metrics_generation(trues[i], preds[i])["bleu"]*100))
