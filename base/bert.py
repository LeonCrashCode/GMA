import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import sys

#datapath = f'/content/drive/My Drive/Medium/bbc-text.csv'
#df = pd.read_csv(datapath)
#df.head()


#df.groupby(['category']).size().plot.bar()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

labels = {
        "NO":0,
        "ideational":1,
        "interpersonal-mood-offer":2,
        "interpersonal-mood-command":3,
        "interpersonal-mood-statement":4,
        "interpersonal-mood-question":5,
        "interpersonal-modality-probability":6,
        "interpersonal-modality-usuality":7,
        "interpersonal-modality-obligation":8,
        "interpersonal-modality-inclination":9
        }

label_list = [
        "NO",
        "ideational",
        "interpersonal-mood-offer",
        "interpersonal-mood-command",
        "interpersonal-mood-statement",
        "interpersonal-mood-question",
        "interpersonal-modality-probability",
        "interpersonal-modality-usuality",
        "interpersonal-modality-obligation",
        "interpersonal-modality-inclination"
        ]

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df[1]]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 50, truncation=True,
                                return_tensors="pt") for text in df[0]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 10)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)
    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=10)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr= learning_rate)

    if use_cuda:

            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train): .3f} | Train Accuracy: {total_acc_train / len(train): .3f} | Val Loss: {total_loss_val / len(val): .3f} | Val Accuracy: {total_acc_val / len(val): .3f}')
                  

def evaluate(model, test_data):

    ret = [[], []]

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              ret[1] += output.argmax(dim=1).to('cpu').tolist()
              ret[0] += test_label.to('cpu').tolist()

              acc = (output.argmax(dim=1) == test_label).sum().item()
              total_acc_test += acc
    
    print(f'Test Accuracy: {total_acc_test / len(test): .3f}')
    return ret

def preprocess(filename):
    
    ret = [[], []]
    for line in open(filename):
        line = line.strip()
        if line == "":
            continue
        line = line.split("|")
        ret[0].append(line[0])
        ret[1].append(line[1])
    return ret

train_file = sys.argv[1]
val_file = sys.argv[2]
test_file = sys.argv[3]
output_file = sys.argv[4]

df_train = preprocess(train_file)
df_val = preprocess(val_file)
df_test = preprocess(test_file)

EPOCHS = 10
model = BertClassifier()
LR = 1e-6
              
train(model, df_train, df_val, LR, EPOCHS)

ret = evaluate(model, df_test)
with open(output_file, "w") as f:
    for a, b in zip(ret[0], ret[1]):
        output = []
        if label_list[a] == "NO":
            output.append("NO /")
        else:
            output.append("YES {}".format(label_list[a]))
        
        if label_list[b] == "NO":
            output.append("NO /")
        else:
            output.append("YES {}".format(label_list[b]))
        f.write(" | ".join(output)+"\n")
    f.close()


