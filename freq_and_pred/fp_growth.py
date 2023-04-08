import numpy as np
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import time

'''
Generate frequent itemset with fp_growth algorithm into csv file
'''

train_file = open('/Users/jiamubai/Documents/prob and stats/prob-and-stats/freq_and_pred/train1000.txt', 'r')
record = []
count = 0
for lines in train_file:
    lines = lines.strip()
    lines = lines.split(':')[1].strip('][').split(', ')
    record.append(lines)


record = np.array(record)
print(record.shape)

# initializing the transactionEncoder
te = TransactionEncoder()
te_ary = te.fit(record).transform(record)
dataset = pd.DataFrame(te_ary, columns=te.columns_)

min_support = 1/466119


#running the fpgrowth algorithm
start = time.time()
res=fpgrowth(dataset,min_support=min_support, use_colnames=True)
end = time.time()
# res.to_csv('freq_set_fp_1000_conf_div_75.csv')
print('Time:', end - start)

res.to_csv('freq_set_fp_1000.csv')

