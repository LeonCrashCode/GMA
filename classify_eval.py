from utils_metrics import metrics_classification
import sys
trues_list = []
preds_list = []

for line in open(sys.argv[1]):
    line = line.strip().split("|")
    trues_list.append(line[0].split())
    preds_list.append(line[1].split())


scores = metrics_classification(trues_list, preds_list)

#for key in scores.keys():
#    print(key, scores[key])

print("====acc: ", scores["acc"])

print("====precision")
p = scores["precision"]
for key in p.keys():
    print("{}: {}".format(key, p[key]))

print("====recall")
r = scores["recall"]
for key in r.keys():
    print("{}: {}".format(key, r[key]))

print("====F1-score")
f = scores["f1-score"]
for key in f.keys():
    print("{}: {}".format(key, f[key]))

print("====macro average f1-score: ", scores["macro-f"])
print("====micro average f1-score: ", scores["micro-f"])
