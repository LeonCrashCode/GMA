from utils_metrics import metrics_classification 
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import time
import math

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def make_labels(score, label_list):
   
    binary_score = score[0:2]
    multiple_score = score[2:]

    label_index = binary_score.index(max(binary_score))
    binary_label = label_list[label_index]

    label_index = multiple_score.index(max(multiple_score)) + 2
    multiple_label = label_list[label_index]

    if binary_label == "NO":
        return [binary_label, "/"]
    else:
        return [binary_label, multiple_label]

def prediction(input_TXT):

    model.to(device)

    template_list = [
        "It is a straightforward expression",
        "It is a grammatical metaphorical expression",
        "It is decorated with interpersonal-mood grammatical metaphor, expressing a offer",
        "It is decorated with interpersonal-mood grammatical metaphor, expressing a command",
        "It is decorated with interpersonal-mood grammatical metaphor, expressing a statement",
        "It is decorated with interpersonal-mood grammatical metaphor, expressing a question",
        "It is decorated with interpersonal-modality grammatical metaphor, expressing a probability",
        "It is decorated with interpersonal-modality grammatical metaphor, expressing a usuality",
        "It is decorated with interpersonal-modality grammatical metaphor, expressing a obligation",
        "It is decorated with interpersonal-modality grammatical metaphor, expressing a inclination",
        "It is decorated with ideational grammatical metaphor"
        ]

    label_list = [
        "NO",
        "YES",
        "interpersonal-mood-offer",
        "interpersonal-mood-command",
        "interpersonal-mood-statement",
        "interpersonal-mood-question",
        "interpersonal-modality-probability",
        "interpersonal-modality-usuality",
        "interpersonal-modality-obligation",
        "interpersonal-modality-inclination",
        "ideational"
        ]

    template_list_length = len(template_list)

    input_TXT = [input_TXT] * template_list_length
    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']

    output_ids = tokenizer(template_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
     
    output_length_list = []
    for i in range(template_list_length):
        output_length = tokenizer(template_list[i], return_tensors='pt', padding=True, truncation=True)['input_ids'].shape[1] - 2
        output_length_list.append(output_length)

    score = [1] * template_list_length
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]        
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            logits = logits.to('cpu').numpy()
            for j in range(template_list_length):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]


    pred_labels = make_labels(score, label_list)

    return pred_labels

def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday ."
#model = BartForConditionalGeneration.from_pretrained('./checkpoint-3060')
model = BartForConditionalGeneration.from_pretrained('./outputs/best_model/')
# model = BartForConditionalGeneration.from_pretrained('../dialogue/bart-large')
model.eval()
model.config.use_cache = False
# input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
# print(input_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_file = sys.argv[1]
output_file = sys.argv[2]
examples = []
for line in open(file_path):
    line = line.strip()
    if line == "":
        continue
    words, labels, targets = line.split("\t")
    InputExample(words=words, labels=labels.split())
    examples.append(InputExample)    

trues_list = []
preds_list = []
num_01 = len(examples)
num_point = 0
start = time.time()
for example in examples:
    sources = example.words
    preds_list.append(prediction(sources))
    trues_list.append(example.labels)
    print('%d/%d (%s)'%(num_point+1, num_01, cal_time(start)))
    num_point += 1

with open(output_file, "w") as f:
    for a, b in zip(trues_list, preds_list):
        f.write("{} | {}\n".format(" ".join(a), " ".join(b)))
    f.close()

scores = metrics_classification(trues_list, preds_list)
