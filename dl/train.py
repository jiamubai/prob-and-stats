import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from utils import *
import json 
from torch.utils.data import TensorDataset, random_split, RandomSampler, SequentialSampler, DataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random 
from tqdm import tqdm
from sklearn.metrics import top_k_accuracy_score
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--epochs", type=int, default=4, help="Number of epochs to train for.")
parser.add_argument("--distinct_count", type=int, default=500, help="Number of distinct samples to generate.")
parser.add_argument("--random_sample", default=False,type=bool, help="Whether to generate a random sample of the dataset.")

args = parser.parse_args()

print("Epochs:", args.epochs)
print("Distinct count:", args.distinct_count)
print("Random sample:", args.random_sample)

### Parameters to change
epochs = args.epochs
distinct_count = args.distinct_count
random_sample = args.random_sample
batch_size = 32

sampling = 'random' if random_sample else 'frequent'

#train data
save_path = '/scratch/ds5749/prob-and-stats/dl/'
train_data = read_file('train.txt')
train_data_int = [ [int(i[1:-1]) for i in line]  for line in train_data ]

#get names of the id
products = pd.read_csv('/scratch/ds5749/prob-and-stats/products.csv')
id2name_mapping = dict(zip(products['product_id'], products['product_name']))
name2id_mapping = {value: key for key, value in id2name_mapping.items()}
train_data_names = [ [id2name_mapping[i] for i in line]  for line in train_data_int ]


items, data = filter_file(train_data_names, n_items=distinct_count, random_sample=random_sample)

#save data mapping
flat_list =  [item for sublist in data for item in sublist]
mapping = {item: i for i, item in enumerate(set(flat_list))}
with open(f"{save_path}/{sampling}_item_map_{distinct_count}.json", "w") as outfile:
    json.dump(mapping, outfile)
    
#test data
test_data = read_file('eval.txt')
test_data_int = [ [int(i[1:-1]) for i in line]  for line in test_data ]
train_data_names = [ [id2name_mapping[i] for i in line]  for line in test_data_int ]
_, test_data = filter_file(train_data_names, n_items=distinct_count)
for i in range(len(test_data)):
    test_data[i] = [item for item in test_data[i] if item in items]
test_data = [cart for cart in test_data if len(cart) > 1]

    
from transformers import AutoTokenizer, BertTokenizer, AutoModelForSequenceClassification
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           num_labels= distinct_count,
                                                           output_attentions = False,
                                                          output_hidden_states = False)
model.to(device)
model.train()

#get max token length
max_len= 0
for sent in data:
    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    max_len = max(max_len, len(input_ids))
print('Max sentence length: ', max_len)
        
    
def generate_dataset(df, tokenizer, mapping, max_len=512, shuffle=False):
    input_ids_list, labels = [], []
    attention_masks_list = []
    for index in tqdm(range(len(df))):
        label = df[index][-1]
        text = df[index][0:-1]
        if shuffle:
            random.shuffle(text)

        text = ','.join(text)
        encoded_input = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        pad_to_max_length = True,
                        max_length = max_len,     
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation=True,
                        return_tensors = 'pt',
                   )
        # input_ids, attention_mask = torch.tensor(encoded_input['input_ids']), torch.tensor(encoded_input['attention_mask'])
        one_hot_encoded_label = torch.tensor([0] * len(mapping))
        one_hot_encoded_label[mapping[str(label)]] = 1
        
        input_ids_list.append(encoded_input['input_ids'])
        attention_masks_list.append(encoded_input['attention_mask'])
        labels.append(one_hot_encoded_label.unsqueeze(0))
    
    
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    labels = torch.cat(labels, dim=0)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset

# dataset = generate_dataset(data, tokenizer, mapping, max_len=147, shuffle=False)
# train_size = int(0.9 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset = generate_dataset(data, tokenizer, mapping, max_len=max_len, shuffle=True)
val_dataset = generate_dataset(test_data, tokenizer, mapping, max_len=max_len, shuffle=False)

print('{:>5,} training samples'.format(len(train_dataset)))
print('{:>5,} validation samples'.format(len(val_dataset)))


train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

from transformers import get_linear_schedule_with_warmup, AdamW

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
criterion = torch.nn.CrossEntropyLoss()


training_stats = []

for epoch_i in range(0, epochs):
    total_train_loss = 0
    model.train()
    for i, batch in tqdm(enumerate(train_dataloader)):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask ,labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        model.zero_grad()  
        
        outs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits =  outs.logits
        loss = criterion(logits,labels.float())

        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
     
    avg_train_loss = total_train_loss / len(train_dataloader)     
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    print("")
    print("Running Validation...")
    model.eval()
    top5, top10, top20 = 0,0,0
    total_eval_loss = 0
    for batch in tqdm(validation_dataloader):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask ,labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        with torch.no_grad():        
            outs = model(input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask)
            logits =  outs.logits
            loss = criterion(logits,labels.float())
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = torch.argmax(labels, dim=-1).to('cpu').numpy()
        
        top5 += top_k_accuracy_score(label_ids,logits, k=5,labels=range(distinct_count))
        top10 += top_k_accuracy_score(label_ids,logits, k=10,labels=range(distinct_count))
        top20 += top_k_accuracy_score(label_ids,logits, k=10,labels=range(distinct_count))
    top5 = top5 / len(validation_dataloader)
    top10 = top10 / len(validation_dataloader)
    top20 = top20 / len(validation_dataloader)
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    print("  Accuracy@5: {0:.5f}".format(top5))
    print("  Accuracy@10: {0:.5f}".format(top10))
    print("  Accuracy@20: {0:.5f}".format(top20))
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accuracy@5.': top5,
            'Valid. Accuracy@10.': top10,
            'Valid. Accuracy@20.': top20,
        }
    )

with open(f'/scratch/ds5749/prob-and-stats/dl/models/{sampling}_{distinct_count}_model_epoch_{i}_result.json', 'w') as f:
    json.dump(training_stats, f, indent=4) 
    
    
    
def generate_dataset(df, tokenizer, mapping, max_len=512, shuffle=False):
    input_ids_list, labels = [], []
    attention_masks_list = []
    for index in tqdm(range(len(df))):
        label = df[index][-1]
        text = df[index][0:-1]
        if shuffle:
            random.shuffle(text)
        text = ','.join(text)
        encoded_input = tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        pad_to_max_length = True,
                        max_length = max_len,     
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation=True,
                        return_tensors = 'pt',
                   )
        # input_ids, attention_mask = torch.tensor(encoded_input['input_ids']), torch.tensor(encoded_input['attention_mask'])
        one_hot_encoded_label = torch.tensor([0] * len(mapping))
        one_hot_encoded_label[mapping[str(label)]] = 1
        
        input_ids_list.append(encoded_input['input_ids'])
        attention_masks_list.append(encoded_input['attention_mask'])
        labels.append(one_hot_encoded_label.unsqueeze(0))
    
    
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    labels = torch.cat(labels, dim=0)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset