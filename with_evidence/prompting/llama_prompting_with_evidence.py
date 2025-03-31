import requests
import pickle
import ast
import csv
import pandas as pd
import ast
import re


def with_evidence_llm(claim, evidence):
    # This function is to create instruction and content for the llm verification
    instruction = (
        "You are an advanced AI fact-checker trained on a vast corpus, including information from Wikipedia. "
        "You are given a claim and evidence, and your task is to verify whether all the facts in the claim are supported by "
        "the evidence and your knowledge base. Use your understanding of evidence and the world, and factual data to make an accurate judgment. "
        "Your response must follow the given format exactly as specified below, without any deviation. "
        "### Response Format:"
        "Your response should be a True/False exactly either True or False"
    )

    content = f'''
    ## TASK:
    Verify the following claim using your knowledge and the given evidence:

    Claim: {claim}

    Evidence: {evidence}

    ### Instructions:
    Assess whether the claim is true or false based on the given evidence and your knowledge.

    ### Answer Format:
    Provide your response as a boolean answer:
    "True" or "False" (single word answer)

    ### Example 1:
    Claim: "The Earth is flat."
    Response: "False"

    ### Example 2:
    Claim: "Albert Einstein developed the theory of relativity."
    Response: "True"
    '''
    return instruction, content

# Cloudflare API tokens and URL for inference
cloudfare_api = "Bearer cloudfare_api" # exactly replace cloudfare_api with actual api, keeping Bearer and space before it
account_id = "account_id"

API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
headers = {"Authorization": cloudfare_api}

def run(model, inputs):
    # To send API request to run for LLM inference
    input = { "messages": inputs }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

def send_request(instruction, content):
    # Here we actually prepare the input and send it for LLM inference/verification
    inputs = [
        { "role": "system", "content": instruction},
        { "role": "user", "content": content}
    ];
    output = run("@cf/meta/llama-3.2-3b-instruct", inputs)
    print(output)
    response = output['result']['response']
    return response
    

def str_to_bool_str(response):
    # this function is to convert the response to a boolean from a string value
    try:
        if isinstance(response, str):
            # Extract only the first occurrence of "true" or "false" (case insensitive)
            match = re.search(r'\b(true|false)\b', response, re.IGNORECASE)
            if match:
                return [match.group(1).lower() == "true"]  # Convert to boolean list
        return False
    except Exception as e:
        print(f"Error processing response string: {e}")
        return False

def get_reasoning_type(claim, test_data):
    # get the reasoning type for the given claim
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


def write_to_csv(filename, claim, reasoning_type, true_label, predicted_label):
    # to write the results to a CSV file
    fieldnames = ["claim", "reasoning_type", "true_label", "predicted_label"]
    
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writeheader()
        
        writer.writerow({"claim": claim, "reasoning_type": reasoning_type, "true_label": true_label, "predicted_label": predicted_label})

# here two files make up the whole test set;
# with_evidence/prompting/test_with_evi_all.pickle
# with_evidence/prompting/test_with_evi_small.pickle
with open('with_evidence/prompting/test_with_evi_small.pickle', 'rb') as file:
    data = pickle.load(file)

# iterate through claims in the test set and write the results to a CSV file
for claim, information in data.items():
    print(information)
    evidence = information['filtered']
    instruction, content = with_evidence_llm(claim, evidence)
    response = send_request(instruction, content)

    response = str_to_bool_str(response) # try to convert the response into an actual list
    while not response: # if not in the correct format
        response = send_request(instruction, content)
        response = str_to_bool_str(response)

    true_label = information['label'][0]
    predicted_label = response[0]
    reasoning_type = information['reasoning_types'][0]
    PATH_TO_RESULT_FILE = "with_evidence/prompting/prompting_with_evidence_new.csv"
    write_to_csv(PATH_TO_RESULT_FILE, claim, reasoning_type, true_label, predicted_label)
