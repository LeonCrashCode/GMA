from utils_metrics import metrics_generation
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import time
import math
import sys

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def prediction(input_TXT):

    model.to(device)

    template = "Straightforwardly speaking,"

    input_TXT += input_TXT + " " + template

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']

    """
    output_ids = tokenizer(template, return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
    
    print(input_ids.shape)
    print(output_ids.shape)

    print(output_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 1].to(device))[0]        
        print(output.shape)
        index = output.argmax(dim=-1)[:,:MAX_LENGTH].to('cpu')
        print(index.shape)
        print(index)
        generated_output = tokenizer.batch_decode(index, skip_special_tokens=True)  
        print(generated_output) 
    return generated_output[0]
    """

    output_ids = model.generate(input_ids.to(device), num_beams=1, min_length=0, max_length=MAX_LENGTH)
    generated_output = tokenizer.batch_decode(output_ids.to('cpu'), skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return generated_output[0]

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
MAX_LENGTH = 50
examples = []
for line in open(input_file):
    line = line.strip()
    if line == "":
        continue
    words, labels, targets = line.split("|")
    examples.append(InputExample(words=words, labels=targets))

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
        f.write("{} | {}\n".format(a, b))
    f.close()

scores = metrics_generation(trues_list, preds_list)
