from tqdm import tqdm


file = open('order_products__prior.csv', 'r')
file = open('order_products__train.csv', 'r')

order_id = -99
items = []
with open("train.txt", 'w') as f:
    for lines in tqdm(file):
        lines = lines.strip()
        lines = lines.split(',')
        if order_id != lines[0]:
            if len(items) > 1:
                f.write('%s:%s\n' % (order_id, items))
            order_id = lines[0]
            items = []
        items.append(lines[1])
    f.write('%s:%s\n' % (order_id, items))
    
items = []
with open("eval.txt", 'w') as f:
    for lines in tqdm(file):
        lines = lines.strip()
        lines = lines.split(',')
        if order_id != lines[0]:
            if len(items) > 1:
                f.write('%s:%s\n' % (order_id, items))
            order_id = lines[0]
            items = []
        items.append(lines[1])
    f.write('%s:%s\n' % (order_id, items))
