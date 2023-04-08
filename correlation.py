import json
import numpy as np

class correlation_rec:
    def __init__(self, path):
        self.file = self.read_file(path)

    def read_file(self, path):
        file = open(path, 'r')
        record = []
        for lines in file:
            lines = lines.strip()
            lines = lines.split(':')[1].strip('][').split(', ')
            record.append(lines)
        return record

    def filter_file(self, n_items=None, min_frequency=None):
        if n_items is None and min_frequency is None:
            raise ValueError("Both parameters cannot be None")
        elif n_items is not None and min_frequency is not None:
            raise ValueError("Only one parameter can be specified")
        else:
            self.items = {}
            for line in record:
                for key in line:
                    self.items[key] = self.items.get(key, 0) + 1
            self.item_frequency = {v: k for k, v in self.items.items()}
            if n_items is not None:
                self.items = set([self.item_frequency[key] for key in sorted(self.item_frequency)[-(n_items+1):-1]])
            elif min_frequency is not None:
                self.items = set([self.item_frequency[key] for key in sorted(self.item_frequency) if key > min_frequency])
            for i in range(len(self.file)):
                self.file[i] = [item for item in self.file[i] if item in self.items]
            self.file = [cart for cart in self.file if len(cart) > 1]

    def corr_matrix(self, save_path=None):
        orders = self.file
        items = sorted(set(item for order in orders for item in order))
        self.items_name_map = {i: items[i] for i in range(len(items))}
        self.name_items_map = {v: k for k, v in self.items_name_map.items()}

        # create a binary matrix where each element is 1 if the item is in the order, and 0 otherwise
        binary_matrix = np.zeros((len(orders), len(items)))
        for i, order in enumerate(orders):
            for j, item in enumerate(items):
                if item in order:
                    binary_matrix[i, j] = 1
                    
        self.corr_matrix = np.corrcoef(binary_matrix, rowvar=False)
        if save_path is not None:
            np.save(f"{save_path}/corr_matrix_{len(self.items)}.npy", self.corr_matrix)
            with open(f"{save_path}/item_map.json", "w") as outfile:
                json.dump(self.items_name_map, outfile)
        return self.corr_matrix, self.items_name_map
                                   
    def load_corr_matrix(self, corr_matrix_path, items_path):
        self.corr_matrix = np.load(corr_matrix_path)
        with open(items_path, "r") as infile:
                self.items_name_map = json.load(infile)
                
    def predict(self, x, n_output):
        x = [self.name_items_map[item] for item in x if x in self.name_items_map]
        sum_corr = [0]*len(self.name_items_map)
        for i in range(self.corr_matrix.shape[0]):
            for j in range(i, self.corr_matrix.shape[0]):
                    sum_corr[j] += self.corr_matrix[i,j]
        index = np.argsort(arr)[-n_output:]
        pred = [self.items_name_map[i] for i in index]
        return pred 