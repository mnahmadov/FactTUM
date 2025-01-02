import csv
import pickle
import pandas as pd

initial_file = 'claim_only/first_iteration.csv'
with_reasoning_file = 'claim_only/first_iteration_v1.csv'

def write_to_csv(filename, claim, reasoning_type, true_label, predicted_label):
    fieldnames = ["claim", "reasoning_type", "true_label", "predicted_label"]
    
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writeheader()
        
        writer.writerow({"claim": claim, "reasoning_type": reasoning_type, "true_label": true_label, "predicted_label": predicted_label})

with open('claim_only/factkg_dataset/factkg_test.pickle', 'rb') as file:
    # there is this 946th claim that contains some harmful information that llm refuse to answer
    test_data = pickle.load(file)

def get_reasoning_type(claim):
    tags = test_data[claim]['types']
    resoning_types = []
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
    return resoning_types



without_reasoning_types = pd.read_csv(initial_file, dtype={'true_output': bool, 'predicted_output': bool})

for i in range(len(without_reasoning_types)):
    claim = without_reasoning_types.iloc[i]['claim']
    
    reasoning_types = get_reasoning_type(claim)
    if len(reasoning_types) != 1:
        print("Something is odd")
        print("Claim: ", claim)
        print("Reasoning types: ", reasoning_types)
        break

    reasoning_type = reasoning_types[0]
    true_label = without_reasoning_types.iloc[i]['true_output']
    predicted_label = without_reasoning_types.iloc[i]['predicted_output']
    write_to_csv(with_reasoning_file, claim, reasoning_type, true_label, predicted_label)


