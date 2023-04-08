import numpy as np
from tqdm import tqdm
class eval:
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
            
    def eval_data(self, pred_pct):
        import random
        self.x = []
        self.labels = []
        for cart in self.file:
            n = max(int(len(cart)*pred_pct), 1)
            random_items = random.sample(cart, n)
            self.x.append([item for item in cart if item not in random_items])
            self.labels.append(random_items)
        return self.x, self.labels
    
    def accuracy(self, preds, labels):
        acc = []
        for i in range(len(labels)):
            label = set(labels[i])
            acc.append(sum([1 for pred in preds[i] if pred in label])/len(label))
        return np.mean(acc), np.mean([1 if m > 0 else 0 for m in acc]), acc
    
    def evalute(self, preds):
        # from sklearn.metrics import precision_recall_fscore_support
        # labels = np.array(self.x).flatten()
        # preds = labels = np.array(preds).flatten()
        # return precision_recall_fscore_support(self.labels, preds)
        return self.accuracy(preds, self.labels)
    
    def confidence_interval(self, data):
        from scipy import stats
        mean = np.mean(data)
        std_dev = np.std(data)

        # calculate the 95% confidence interval
        conf_interval = stats.t.interval(0.95, len(data)-1, loc=mean, scale=std_dev/len(data)**0.5)
        return mean, conf_interval

    def mc_evalute(self, pred_pct, n, n_output, predictor):
        #evalute using confidence interval
        metrics = []
        for i in tqdm(range(n)):
            x, y = self.eval_data(pred_pct)
            output = []
            for inputs in x:
                out = predictor(inputs, n_output)
                output.append(out)
            metric = self.evalute(output)
            metrics.append(metric[1])
        return self.confidence_interval(metrics)     
    
def plot_hist_metrics(metrics):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.displot(, kde=True, bins=15)
    plt.show()
    
    
def random_predictor(x, n_out):
    n_out = 50
    out = random.sample(validation.items, n_outs)
    return out

'''
validation = eval('eval.txt')
validation.filter_file(n_items=500)
# validation.filter_file(min_frequency=300)
x, y = validation.eval_data(0.2)

n_outs = 50
output = []
for input in x:
    ## algorithm
    out = random.sample(validation.items, n_outs)
    ##
    output.append(out) 
metrics = validation.evalute(output)

validation.mc_evalute(0.2, 100, 50, random_predictor)

'''