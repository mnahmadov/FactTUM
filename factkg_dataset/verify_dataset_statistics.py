# This script is used to verify reasoning type retrival approach
# It was used on the whole FactKG dataset and verified results by comparing statistics 
# to the dataset statistics on the original FactKG paper.

import pickle

data_files = {
    'test': 'claim_only/factkg_dataset/factkg_test.pickle',
    'train': 'claim_only/factkg_dataset/factkg_train.pickle',
    'dev': 'claim_only/factkg_dataset/factkg_dev.pickle'
}
data = {key: pickle.load(open(path, 'rb')) for key, path in data_files.items()}

reasoning_types = {'one-hop': 0, 'conjuction': 0, 'existence': 0, 'multi hop': 0, 'negation': 0}

def update_reasoning_types(dataset):
    for information in dataset.values():
        for tag in information['types']:
            if tag == 'negation':
                reasoning_types['negation'] += 1
                break
            elif tag == 'num1':
                reasoning_types['one-hop'] += 1
            elif tag == 'multi claim':
                reasoning_types['conjuction'] += 1
            elif tag == 'existence':
                reasoning_types['existence'] += 1
            elif tag == 'multi hop':
                reasoning_types['multi hop'] += 1

for dataset in data.values():
    update_reasoning_types(dataset)

reasoning_types['total'] = sum(reasoning_types.values())

print(reasoning_types)