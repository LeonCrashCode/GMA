from collections import defaultdict
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

def metrics_generation(trues_list, preds_list):

    references = []
    candidates = []
    for a, b in zip(trues_list, preds_list):
        references.append([a.split()])
        candidates.append(b.split())

    return {'bleu': corpus_bleu(references, candidates)}

def metrics_classification(trues_list, preds_list):
    acc = 0.0
    for a, b in zip(trues_list, preds_list):
        if a[0] == b[0]:
            acc += 1

    acc /= len(trues_list)
  
   
    mul_acc = defaultdict(float)
    mul_t = defaultdict(float)
    mul_p = defaultdict(float)

    for item in trues_list:
        mul_acc[item[1]] = 0.0
        mul_t[item[1]] = 0.0
        mul_p[item[1]] = 0.0
    mul_acc["YES"] = 0.0
    mul_t["YES"] = 0.0
    mul_p["YES"] = 0.0


    for label in mul_acc.keys():
        if label == "/":
            continue
        for a, b in zip(trues_list, preds_list):
            if a[1] == label:
                mul_t[label] += 1
            if b[1] == label:
                mul_p[label] += 1
            if a[1] == b[1] and a[1] == label:
                mul_acc[label] += 1
            
            if a[0] == label:
                mul_t[label] += 1
            if b[0] == label:
                mul_p[label] += 1
            if a[0] == b[0] and a[0] == label:
                mul_acc[label] += 1
            

                
    p = defaultdict(float)
    r = defaultdict(float)
    f = defaultdict(float)

    for label in mul_acc.keys():
        if label == "/":
            continue
        p[label] = mul_acc[label] / mul_p[label] if mul_p[label] != 0 else 0
        r[label] = mul_acc[label] / mul_t[label] if mul_t[label] != 0 else 0
        f[label] = 2 * p[label] * r[label] / (p[label] + r[label]) if p[label] + r[label] != 0 else 0


    macrof = 0
    cnt = 0
    for label in f.keys():
        if label == "YES":
            continue
        macrof += f[label]
        cnt += 1
    macrof /= cnt
    
    tp = 0
    fp = 0
    fn = 0
    for label in f.keys():
        if label == "YES":
            continue
        tp += mul_acc[label]
        fp += mul_p[label] - mul_acc[label]
        fn += mul_t[label] - mul_acc[label]

    microf = tp / (tp + (fp + fn)/2)
   
    return {'acc': acc, 'precision': p, 'recall': r, 'f1-score': f, "macro-f": macrof, "micro-f": microf}

