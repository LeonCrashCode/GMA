from utils_metrics import metrics_generation
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import torch
import time
import math

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def prediction(input_TXT):

    model.to(device)

    template = "Straightforwardly speaking,"

    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']

    output_ids = tokenizer(template.split(), return_tensors='pt', padding=True, truncation=True)['input_ids']
    output_ids[:, 0] = 2
     
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 1].to(device))[0]        
        index = output.argmax(dim=-1)[:,:MAX_LENGTH].to('cpu')
        print(index.shape)
        generated_output = tokenizer.batch_decode(index, skip_special_tokens=True)  

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
MAX_length = 50
examples = []
for line in open(file_path):
    line = line.strip()
    if line == "":
        continue
    words, labels, targets = line.split("\t")
    InputExample(words=words, labels=targets)
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
        f.write("{} | {}\n".format(a, b))
    f.close()

scores = metrics_generation(trues_list, preds_list)
