import random
import pickle
from constants import PATH_DB,PATH_DATA,headers,API_BASE_URL,TRAIN_SPLITS,TEST_SPLITS,VAL_SPLITS
import numpy as np
import pandas as pd
import argparse
from utils import set_global_log_level
from argparse import ArgumentParser
import requests
from multiprocessing import Pool
from functools import partial
import csv
import json
import random
class KG():
    def __init__(self, kg):
        super().__init__()
        self.kg = kg
        
    def get_one_hop_subgraph(self, entity):
        """
        Extract the one-hop subgraph for the given entity.
    
        Args:
            entity (str): The entity for which to extract the subgraph.
    
        Returns:
            dict: A dictionary with keys as relation, containing the respective neighbors.
        """
        subgraph = {}
        if entity in self.kg:
            for rel,tails in self.kg[entity].items():
                subgraph[rel]=tails
            
        return subgraph
def load_db(path):
    """
    Load the knowledge base
    """
    with open(path, 'rb') as file:
        dbpedia = pickle.load(file)
    return dbpedia


def get_one_hop(kg,split,file_num=10):
    """
    Get all one-hop evidence for the split
    
    Args: kg: the knowledge graph
    split: split name
    file_num: split number 
    """
    for i in range(file_num):
        with open(f'./data/{split}_data_part_{i+1}.pickle', 'rb') as file:
            test_pedia = pickle.load(file)
        evidence={}
        for claim, information in test_pedia.items():
            tags=information["types"]
            resoning_types=[]
            for tag in tags:
                if tag == 'negation':
                    resoning_types.append('negation')
                    break
                elif tag == 'num1':
                    resoning_types.append('one-hop')
                elif tag == 'multi claim':
                    resoning_types.append('conjuction')
                elif tag == 'existence':
                    resoning_types.append('existence')
                elif tag == 'multi hop':
                    resoning_types.append('multi hop')
            subevidence = {
        "entity_set":{},
        "evidence_onehop_full":{},
        "label":information["Label"],
        "reasoning_types":resoning_types
            }
            subevidence["entity_set"]=information['Entity_set']
            for entity in information['Entity_set']:
                subevidence["evidence_onehop_full"][entity]=kg.get_one_hop_subgraph(entity)
            evidence[claim]=subevidence

        output_path= f'./data/{split}_evi_part_{i+1}.pickle'   
        with open(output_path, 'wb') as output_file:
            pickle.dump(evidence, output_file)



def process_response(text,information):
    """
    Process the raw response from LLM, identify invalid response for reprocessing
    
    Args: text: raw response in dict format from LLM
    information: original choice for evidence, for identifying invalid response 
    """
    processed_text = text.strip().strip("```python").strip("```").strip()
    processed_text=processed_text.replace("```", "")
    output_dict={}
    valid_output={}
# Convert to a dictionary
    try:
        output_dict = eval(processed_text)
    except Exception as e:
        valid_output={"invalid output"}
    try:
        for entity, relations in output_dict.items():
            if entity in information['entity_set']:
                for relation in relations:
                    if relation in information['evidence_onehop_full'][entity]:
                        valid_output[entity]=relations
    except Exception as e:
        valid_output={"invalid output"}
    return valid_output


def call_llm(claim, entities,evidence):
    """
    Form the messsage to be sent to LLM

    Args: claim: the claim as the context referene for fact-checking
    entities: entities set given with the claim
    evidence: all potential evidence from one-hop neighbors in the kg
    
    Return: message
    """
    entities_filtered = {entity.replace('"', '') for index, entity in enumerate(entities,start=1)}
    output_expectations= "{\n\n" + "".join([f'''"{entity}-{index}": ["..." , "...", ... ],  # options (strictly choose no more than 5 from): ''' + " , ".join(random.sample(list(connections), min(len(connections), 15))) + "\n\n" for index,(entity, connections) in evidence.items()]) + "}"
    
    content = f'''
    Claim:
    {claim}
    '''
    message= [{"role": "system", "content": 
    '''
    You are an intelligent graph relation finder. You are given a single claim and all connections of the entities in the claims, your task is to filter out the connections that are related to the claim that helps fact-checking. "~" beginning connection means reverse connection.'''
 },{"role": "user", "content": content+ '''
    ## TASK:
     - For each of the given entities given below: 
       Filter the connections strictly from the given options that would be relevant to connect given entities to fact-check Claim1.
    - Think clever, there could be multi-step hidden connections, if not direct, that could connect the entities somehow.
    - Arrange them based on their relevance. Be extra careful with ~ signs.
    - No code output. No explanation. Output only valid python DICT of structure:\n'''+ output_expectations}]

    return message

def run(model, inputs):
    input = { "messages": inputs }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

def send_request(inputs):
    """
    sending the message to LLM and return the response
    """
    output = run("@cf/meta/llama-3.2-3b-instruct", inputs)
    response = output['result']['response']
   # print(response)
    return response

def write_to_csv(evidence_filtered, filename):
    """
    Write the evidence_filtered dictionary to a CSV file.
    
    Args:
        evidence_filtered (dict): The dictionary containing claim information.
        filename (str): The path to the CSV file.
    """
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Claim', 'Evidence_Filtered'])
        
        # Write the content
        for claim, information in evidence_filtered.items():
            writer.writerow([
                claim,
                '; '.join(information.get('entity_set', [])),  # Serialize entity_set as a semicolon-separated string
                json.dumps(information.get('evidence_filtered', {}), ensure_ascii=False)  # Serialize evidence_filtered as JSON
            ])


def get_filter_result(dbpedia,split,file_num):
    """
    post-processing the filtered evidence from LLM

    Args: dbpedia: knowledge base
    split: data split name
    file_num: split number of the corresponding split

    Return: the combined evidence set
    
    """
    for i in range(file_num):
        with open(f'./data/{split}set_filtered_{i+1}.pickle', 'rb') as file:
            data_raw = pickle.load(file)
# Process all entities and save two-hop neighbors
        adddirect={}
        for claim, information in data_raw.items():
            information['direct']={}
            information['filtered']={}
            direct_connection_found = False
            for entity in information['entity_set']:
                if entity in dbpedia:
                    information['direct'][entity]={}
                    for relation, neighbours in dbpedia[entity].items():
                        for neighbour in neighbours:
                            if neighbour in information['entity_set']:
                                information['direct'][entity][relation]=neighbours
                                direct_connection_found = True
                else:
                    print(f"{entity} could not be found")
            if 'evidence_filtered' in information:
                for entity_,relations in information['evidence_filtered'].items():
                    information['filtered'][entity_]={}
                    relations=information['evidence_filtered'][entity_]
                    for relation in relations: 
                        if  relation in information['evidence_onehop_full'][entity_]:
                            information['filtered'][entity_][relation]=random.sample(
                            information['evidence_onehop_full'][entity_][relation],  # Source list
                        min(len(information['evidence_onehop_full'][entity_][relation]), 3)  # Max 3 items
                    )
            else:
                information['evidence_filtered']={}
            del information['evidence_onehop_full']
            if not direct_connection_found:
                information['direct'] = None
        
            adddirect[claim]=information
        output_path=f'./filter_evidence/{split}_comm_evi_{i+1}.pickle'
        with open(output_path, 'wb') as output_file:
            pickle.dump(adddirect, output_file)

if __name__ == "__main__":
    dbpedia=load_db(PATH_DB)
    kg=KG(dbpedia)
    set_global_log_level("info")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", choices=["train", "val", "test", "all"], default="all",
                        help="Dataset split to filter evidence for. ")
    args = parser.parse_args()

    if args.dataset_type == "all":
        dataset_types = ["train", "val", "test"]
    else:
        datasety_types = [args.dataset_type]
    for dataset_type in dataset_types:
        if dataset_type=="train":
            file_num=TRAIN_SPLITS
        elif dataset_type=="val":
            file_num=VAL_SPLITS
        else:
            file_num=TEST_SPLITS
        one_hop=get_one_hop(kg,dataset_type,file_num=file_num)
        for j in file_num:
            csv_filename=f'./data/{dataset_type}set_filtered_{j+1}.csv'
            with open(f'./data/{dataset_type}_evi_part_{j+1}.pickle', 'rb') as file:
                data = pickle.load(file)
        #exceed=0
                evidence_filtered={}
                i=0
                for claim, information in data.items():
                    i+=1
                    print(f"Claim {i} Processing")
                    count=0
                    valid_output = False
                    while (not valid_output):
                       message = call_llm(claim, information['entity_set'], information['evidence_onehop_full'])
                       text = send_request(message)
                       output = process_response(text, information)
                       if output != {'invalid output'}:
                            valid_output = True  # Exit the loop if the output is valid
                            information['evidence_filtered']=output
                       else:
                            if count <5:
                                print(f"Invalid output for claim {claim}, retrying...")
                                count+=1
                            else:
                                print(f"Exceed trying time")
                                #exceed+=1
                                break
                    evidence_filtered[claim]=information
    # Call the function to write to CSV after processing each claim
                 #  write_to_csv(evidence_filtered, csv_filename)
                with open(f'./data/{dataset_type}set_filtered_{j+1}.pickle', 'wb') as output_file:
                        pickle.dump(evidence_filtered, output_file)
        get_filter_result(dbpedia=kg,split=dataset_type,file_num=file_num)

    