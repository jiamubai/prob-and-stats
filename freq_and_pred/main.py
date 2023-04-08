import numpy as np
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
import copy
from tqdm import tqdm
import itertools
import warnings

'''
Input preprocessed file
Use fp_growth algorithm to find all frequent itemsets
Make predictions from the frequent itemsets
'''

warnings.filterwarnings('ignore')

def fp_growth(train_file, min_support):
    '''
    :param train_file: txt file that used for find frequent itemsets
    :param min_support: the minimum support required for each itemset.
    Itemsets with support lower than min_support will not be counted
    as frequent itemsets
    :return: all frequent itemsets with their support
    '''
    record = []
    count = 0
    for lines in train_file:
        lines = lines.strip()
        lines = lines.split(':')[1].strip('][').split(', ')
        record.append(lines)
    record = np.array(record)

    # initializing the transactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(record).transform(record)
    dataset = pd.DataFrame(te_ary, columns=te.columns_)
    # find all frequent itemsets
    res = fpgrowth(dataset, min_support=min_support, use_colnames=True)

    # clean the format
    freq_set = list(res['itemsets'])

    for i in range(len(freq_set)):
        freq_set[i] = str(freq_set[i])[12:-3]
        if ',' in freq_set[i]:
            res['itemsets'][i] = [item for item in freq_set[i].strip('\'').split('\'') if item != '", "']
        else:
            res['itemsets'][i] = [freq_set[i].strip('\'')]
    return res


def eval(purchased, top_num, freq_set_data):
    '''
    :param purchased: the purchased products in the order
    :param top_num: list of number, each number will be the length of the prediction list,
    and the prediction list includes the set of items most likely to be purchased based
    on the information from frequent itemsets.
    :param freq_set_data: frequent itemsets data
    :return: pred_total: list of prediction lists; prob_total: list of probability lists,
    each probability list contain probabilities of each item to be purchased
    '''
    pred_total = []
    prob_total = []
    for i in range(len(top_num)):
        pred_total.append(['None'] * top_num[i])
        prob_total.append(np.zeros(top_num[i]))
    for i in range(len(freq_set_data)):
        #check if all purchased items are in the frequent itemset
        check = all(item in freq_set_data['itemsets'][i] for item in purchased)
        if check:
            for j in range(len(top_num)):
                #replace the smallest probability into the conditional probability (support) of the itemset
                if prob_total[j].min() < float(freq_set_data['support'][i]) and len(freq_set_data['itemsets'][i]) > 1:
                    lst = copy.deepcopy(freq_set_data['itemsets'][i])
                    for item in purchased:
                        lst.remove(item)
                    pred_total[j][np.argmin(prob_total[j])] = lst
                    prob_total[j][np.argmin(prob_total[j])] = float(freq_set_data['support'][i])
    return pred_total, prob_total


def pred(pred_data, top_num, freq_set_data):
    '''
    :param pred_data: txt file that used for evaluation
    :param top_num: list of number, each number will be the length of the prediction list,
    and the prediction list includes the set of items most likely to be purchased based
    on the information from frequent itemsets
    :param freq_set_data: frequent itemsets data
    :return: list of accuracy
    '''
    purchased = []
    true = []
    # split the evaluation orders into purchased items and target items
    for lines in pred_data:
        lines = lines.strip().split(':')[1].strip('][').split(', ')
        purchased.append([item.strip('\'') for item in lines[:-1]])
        true.append([lines[-1].strip('\'')])
    acc_lst = np.zeros(3)
    total = len(purchased)
    #calculate accuracy
    for i in tqdm(range(len(purchased))):
        pred_total, prob_total = eval(purchased[i], top_num, freq_set_data)
        for j in range(len(top_num)):
            if all(val == 'None' for val in pred_total[j]):
                # total -= 1
                continue
            else:
                pred_lst = list(itertools.chain(*pred_total[j]))
                if true[i][0] in pred_lst:
                    acc_lst[j] += 1
    return acc_lst / total


if '__main__':
    train_lst = ['train500.txt', 'train1000.txt', 'train2000.txt']
    eval_lst = ['eval500.txt', 'eval1000.txt', 'eval2000.txt']

    train_file = open(train_lst[2], 'r')
    pred_file = open(eval_lst[2], 'r')
    min_support = [1 / 168153, 1 / 466119, 1 / 897506]
    res = fp_growth(train_file, min_support[2])
    acc = pred(pred_file, [5, 10, 20], res)
    print(acc)
