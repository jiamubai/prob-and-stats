import copy

import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm

'''
Make prediction from the frequent itemset and output acc
'''


pred_data = open('/Users/jiamubai/Documents/prob and stats/prob-and-stats/freq_and_pred/eval1000.txt', 'r')
freq_set_data = pd.read_csv('/freq_set_fp_1000_conf_0.3.csv')

freq_set = list(freq_set_data['itemsets'])

for i in range(len(freq_set)):
    freq_set[i] = freq_set[i][12:-3]
    if ',' in freq_set[i]:
        freq_set_data['itemsets'][i] = [item for item in freq_set[i].strip('\'').split('\'') if item!= '", "']
        # print(freq_set[i])
        # break
    else:
        freq_set_data['itemsets'][i] = [freq_set[i].strip('\'')]


purchased = []
true = []
for lines in pred_data:
    lines = lines.strip().split(':')[1].strip('][').split(', ')
    purchased.append([item.strip('\'') for item in lines[:-1]])
    true.append([lines[-1].strip('\'')])

# print(freq_set_data[freq_set_data['itemsets'].isin([purchased[0]])])



def eval(purchased, top_num):
    pred = ['None']*top_num
    prob = np.zeros(top_num)
    for i in range(len(freq_set_data)):
        check = all(item in freq_set_data['itemsets'][i] for item in purchased)
        if check:
            if prob.min() < float(freq_set_data['support'][i]) and len(freq_set_data['itemsets'][i]) > 1:
                lst = copy.deepcopy(freq_set_data['itemsets'][i])
                for item in purchased:
                    lst.remove(item)
                pred[np.argmin(prob)] = lst
                prob[np.argmin(prob)] = float(freq_set_data['support'][i])
    return pred, prob

# check predict acc
acc = 0
total = len(purchased)
for i in tqdm(range(len(purchased))):
    pred, prob = eval(purchased[i],10)
    if all(val == 'None' for val in pred):
        # total -= 1
        continue
    else:
        pred_lst = list(itertools.chain(*pred))
        if true[i][0] in pred_lst:
            acc += 1

acc = acc/total
print(acc)


