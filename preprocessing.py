from tqdm import tqdm
import pandas as pd

def preprocess(data_raw, product_num, clean_data_name):
    order_id = -99
    items = []
    with open(clean_data_name, 'w') as f:
        new = data_raw[data_raw['product_id'].isin(data_raw['product_id'].value_counts()[:product_num])]
        new.to_csv()
        file = pd.read_csv(new)
        for lines in tqdm(file):
            lines = lines.strip()
            lines = lines.split(',')
            if order_id != lines[1]:
                if len(items) > 1:
                    f.write('%s:%s\n' % (order_id, items))
                order_id = lines[1]
                items = []
            items.append(lines[2])

if '__main__':
    data_raw = pd.read_csv('change_to_file_path')
    product_num = 1000
    clean_data_name = 'change_new_file_name'
    preprocess(data_raw, product_num, clean_data_name)