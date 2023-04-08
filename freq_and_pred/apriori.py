from apyori import apriori
import numpy as np
import time
from tqdm import tqdm

train_file = open('train5000.txt', 'r')
# in total 49688 item
record = []
count = 0
for lines in train_file:
    lines = lines.strip()
    lines = lines.split(':')[1].strip('][').split(', ')
    record.append(lines)
    count += 1
    if count == 1000:
        break

min_support = 1/49688
min_support = 1/5000
# 472565 is the most frequent item, and 32434489 is the total order number
max_support = 472565/32434489
max_support = 472565/1606296
min_confidence = min_support/max_support
# lift = 1 means no correlation among items in itemsets,
# > 1 means positive relationships, <1 means negative relationships
lift = 1

#
# record = [['Wine', 'Chips','Bread','Butter','Milk','Apple'],['Wine','Bread','Butter','Milk'],
#           ['NaN','Bread','Butter','Milk'],['Chips','Apple'],
#           ['Wine', 'Chips','Bread','Butter','Milk','Apple'],['Wine', 'Chips','Bread','Butter','Apple'],
#           ['Wine', 'Chips'],['Wine','Bread','Apple'],
#           ['Wine','Bread','Butter','Milk'],['Wine', 'Chips','Bread','Butter','Apple']]


# print(record)

start = time.time()
apriori_rule = apriori(record, min_support=min_support, min_confidence=min_confidence, min_lift=lift, min_length=1)
end = time.time()

print(len(list(apriori_rule)))


# with open("freq_set5000.txt", 'w') as f:
#     for set in apriori_rule:
#         f.write(str(set))
print('Time: ', end - start)
# for item in list(apriori_rule):
#     print(item)