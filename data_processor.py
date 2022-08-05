import json
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset, DatasetDict
import pyarrow as pa

def flatten(l, dimension):
    for i in range(dimension-1):
        l = [item for sublist in l for item in sublist]
    return l

def shorten(d):
    r = {}
    newpar = flatten(d['paragraphs'],2)
    newlab = flatten(d['gold_labels'],2)
#     print(len(newpar))
#     print(len(newlab))
    while len(flatten(newpar,2)) > (512 - max([len(x) for x in newpar])):
        found = False
        for i in range(len(newpar)):
            if len(newpar[i]) + len(flatten(newpar,2)) > 512 and newlab[i] == 0:
                del newpar[i]
                del newlab[i]
                found = True
                break
        if not found:
            for j in range(len(newpar)):
                if newlab[j] == 0:
                    del newpar[j]
                    del newlab[j]
                    break
#         print(len(flatten(newpar,2)))
#         print(len(newpar[2]))
    r['new_paragraphs'] = newpar
    r['new_label'] = newlab
    return r


def generate_rows(s_ds):
    rows = []
    for i in s_ds:
        text = flatten(s_ds[i]['new_paragraphs'],2)
        for j in range(len(s_ds[i]['new_label'])):
            rows.append((text, s_ds[i]['new_paragraphs'][j],s_ds[i]['new_label'][j]))
#         break
    return rows

def tokenize_f(examples):
    tokenized_inputs = tok(
        examples["sentence1"], examples['sentence2'], truncation=True, padding="max_length"
    )
    tokenized_inputs['labels'] = examples['label']
    return tokenized_inputs

# filecontent = ""
# for i in range(1,6):
#     with open(f'../input/indosum/indosum/train.0{i}.jsonl','r') as f:
#     #     data = json.load(f)
#         filecontent += f.read()

# data = {}
# enum = 0
# for line in filecontent.split('\n'):
# #     print(line)
#     try:
#         data[enum] = json.loads(line)
#         enum+=1
#     except:
#         print(line)


# s_ds = {}
# l = len(data.keys())
# for i in range(l):
#     # if i % 1000 == 0:
#     #     print(i)
#     s_ds[i] = shorten(data[i])

# rows = generate_rows(s_ds)

with open('tes.pck','rb') as f:
    rows = pickle.load(f)

x = pd.DataFrame(rows, columns=['sentence1', 'sentence2', 'label'])

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=6)
# ss.split(x, x['label'])
for train_index, test_index in ss.split(x, x['label']):
    print(len(train_index))
    print(len(test_index))
train = x.loc[train_index]
train = train.reset_index()
train['idx'] = train.index
test = x.loc[test_index]
test = test.reset_index()
test['idx'] = test.index


# labelused = 'ment210'
# for i in data:
#     temp = data[i]
#     df.append([[str(i) for i in temp['newtoken']],temp[labelused], temp['postagl']])
# x = pd.DataFrame(df,columns=['tokens',labelused, 'postag'])
# x = x.dropna()
# train = x[:2250]
# test = x[2250:]
# dstp = Dataset.from_pandas(train)
# dsep = Dataset.from_pandas(test)
dst = pa.Table.from_pandas(train)
dse = pa.Table.from_pandas(test)
dsd = DatasetDict({'train' : Dataset(arrow_table=dst), 'test' : Dataset(arrow_table=dse)})

# dsd['train']
# tokenized_ds = dsd.map(tokenize_f)

with open('spc_sum_data.pck','wb') as f:
    pickle.dump(dsd, f)