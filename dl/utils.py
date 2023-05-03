import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random 

class ItemDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=512, shuffle=False):
        self.df = df
        self.shuffle = shuffle
        self.max_len=max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # row = self.df.iloc[index]
        label = self.df[index][-1]
        text = self.df[index][0:-1]
        if self.shuffle:
            random.shuffle(text)
        text = ','.join(text)
        encoded_input = self.tokenizer.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = self.max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation=True,
                   )
        return encoded_input
    
def split_list(list_to_split, split_ratio):
  """Splits a list into two lists, with the first list containing the first `split_ratio` percentage of the elements, and the second list containing the remaining elements.

  Args:
    list_to_split: The list to split.
    split_ratio: The percentage of elements to include in the first list.

  Returns:
    A tuple of two lists, with the first list containing the first `split_ratio` percentage of the elements, and the second list containing the remaining elements.
  """

  split_index = int(len(list_to_split) * split_ratio)
  train = list_to_split[:split_index]
  val = list_to_split[split_index:]

  return train, val

def read_file(path):
    file = open(path, 'r')
    record = []
    for lines in file:
        lines = lines.strip()
        lines = lines.split(':')[1].strip('][').split(', ')
        record.append(lines)
    return record
train_data = read_file('train.txt')

def filter_file(file, n_items=None, random=False):
    items = {}
    for line in file:
        for key in line:
            items[key] = items.get(key, 0) + 1
    if n_items is not None:
        if random == True:
            keys = list(items.keys())
            items = set([item_frequency[key] for i in np.random.choice(len(keys), n_items, replace=False)])
        else:
            sorted_dictionary = sorted(items.items(), key=lambda x: x[1], reverse=True)
            items = set([x[0] for x in sorted_dictionary[:n_items]])
    for i in range(len(file)):
        file[i] = [item for item in file[i] if item in items]
    file = [cart for cart in file if len(cart) > 1]
    return items, file


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.sum(predictions==labels)/len(labels)
    return accuracy

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def accuracy(self, preds, labels):
    acc = []
    for i in range(len(labels)):
        label = set(labels[i])
        acc.append(sum([1 for pred in preds[i] if pred in label])/len(label))
    return np.mean(acc), np.mean([1 if m > 0 else 0 for m in acc]), acc
