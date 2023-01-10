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

