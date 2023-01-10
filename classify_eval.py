from utils_metrics import metrics_classification

trues_list = []
preds_list = []

for line in open(sys.argv[1]):
    line = line.strip().split("|")
    trues_list.append(line[0].split())
    preds_list.append(line[1].split())


scores = metrics_classification(trues_list, preds_list)
print(scores)
