import random
import pickle
import argparse
from utils import set_global_log_level
from argparse import ArgumentParser
from multiprocessing import Pool
from functools import partial
import random
from constants import TRAIN_SPLITS,TEST_SPLITS,VAL_SPLITS
def get_samples(num=20000):
    """
    Sample from the training and validation set
    
    Args: 
    num: set value for sampling from training set, validation set will be as the same ratio
    """
    with open(f'../../factkg_dataset/factkg_train.pickle', 'rb') as file:
        data_raw = pickle.load(file)
    random.seed(42)  # Set a seed for reproducibility
    selected_keys = random.sample(list(data_raw.keys()), num)
    selected_data = {key: data_raw[key] for key in selected_keys}
    part_size = len(selected_data) //TRAIN_SPLITS  # Size of each part (2,000 in this case)
    parts = []
    keys = list(selected_data.keys())
    start = 0
    for i in range(TRAIN_SPLITS):
        end = start + part_size
    # Handle any leftover keys in the last part
        if i == TRAIN_SPLITS-1:
           end = len(selected_data)
        part = {key: selected_data[key] for key in keys[start:end]}
        parts.append(part)
        start = end

# Save each part into separate pickle files
    for i, part in enumerate(parts):
        with open(f'./data/train_data_part_{i+1}.pickle', 'wb') as file:
            pickle.dump(part, file)

    with open(f'../../factkg_dataset/factkg_train.pickle', 'rb') as file:
        data_raw = pickle.load(file)
    random.seed(42)  # Set a seed for reproducibility
    num_val=num/86367*13266
    selected_keys = random.sample(list(data_raw.keys()), num_val)
    selected_data = {key: data_raw[key] for key in selected_keys}
    part_size = len(selected_data) //VAL_SPLITS  # Size of each part (2,000 in this case)
    parts_val = []
    keys = list(selected_data.keys())
    start = 0
    for i in range(VAL_SPLITS):
        end = start + part_size
    # Handle any leftover keys in the last part
        if i == VAL_SPLITS-1:
           end = len(selected_data)
        part_val = {key: selected_data[key] for key in keys[start:end]}
        parts_val.append(part_val)
        start = end

# Save each part into separate pickle files
    for i, part in enumerate(parts_val):
        with open(f'./data/val_data_part_{i+1}.pickle', 'wb') as file:
            pickle.dump(part, file)

if __name__ == "__main__":
    set_global_log_level("info")

    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size_training", type=int,default=20000,
                        help="Number of file for each split")
    args = parser.parse_args()
    num=args.sample_size_training
    samples=get_samples(num)