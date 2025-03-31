import pandas as pd
import argparse
from utils import set_global_log_level
from pathlib import Path
from constants import DATA_PATH,SUBGRAPH_FOLDER
import os
import pickle
from constants import TRAIN_SPLITS,TEST_SPLITS,VAL_SPLITS
def convert_subgraphs(dic,method):
    """
    Convert the evidence set ino the triple format for pygeomrtric package

    Args: dic: the claim and the corresponding evidence set
    method: chosen from "direct","combined", default set to combined
    
    Return: dict of all the triples

    """
    subgraphs={}
    for index, (claim, information) in enumerate(dic.items()):
        if method=='direct':
            subgraph=[]
            for entity,relations in information['direct'].items():
                for relation,neighbours in information['direct'][entity].items():
                    for neighbour in neighbours:
                        subgraph.append([entity,relation,neighbour])
        if method=='combined':
            subgraph=[]
            if information['direct'] != None:
                for entity,relations in information['direct'].items():
                    for relation,neighbours in information['direct'][entity].items():
                        for neighbour in neighbours:
                            subgraph.append([entity,relation,neighbour])
            if 'filtered' in information:
                for entity_,relations_ in information['filtered'].items():
                    for relation_,neighbours_ in information['filtered'][entity_].items():
                        for neighbour_ in neighbours_:
                            subgraph.append([entity_,relation_,neighbour_])
                        
        else:
            break
        subgraphs[index]=subgraph
    return subgraphs

def save_subgraphs(split,method):
    if split=="train":
        sub_path="train"
        num=TRAIN_SPLITS
    elif split=="val":
        sub_path="dev"
        num=TEST_SPLITS
    elif split=="test":
        sub_path="test"
        num=VAL_SPLITS
    else:
        raise ValueError("Invalid split name")
    combined_dict={}
    for i in range(num):
        with open(f'./filter_evidence/{sub_path}_comm_evi_{i+1}.pickle', 'rb') as file:
            dict = pickle.load(file)
            combined_dict.update(dict)
    subgraphs=convert_subgraphs(combined_dict,method)
    save_path = f'./data/{split}set_subgraph_{method}.pickle'
    subgraph_df = pd.DataFrame()
    subgraph_df['subgraph'] = subgraphs
    subgraph_df.to_pickle(save_path)

def get_subgraph(split,method):
    file_path = f'./data/{split}set_subgraph_{method}.pickle'
    df=pd.read_pickle(file_path)
    return df['subgraph']

if __name__ == "__main__":
    set_global_log_level("info")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to make subgraph for. ")
    parser.add_argument("--method", choices=["direct", "combined","filtered"], default="combine",
                        help="Use `direct` subgraph (only to other entities) or `one-hop` (to any other node). ")
    args = parser.parse_args()

    if args.dataset_type == "all":
        dataset_types = ["train", "val", "test"]
    else:
        datasety_types = [args.dataset_type]

    save_folder = Path(DATA_PATH) / SUBGRAPH_FOLDER
    os.makedirs(save_folder, exist_ok=True)
    for dataset_type in dataset_types:
        save_subgraphs(dataset_type,args.method)
  