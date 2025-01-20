import requests
import pickle
import ast
import csv
import pandas as pd

def shorten_evidence(claim, evidence):
    instruction = (
        "You are an intelligent graph connection finder."
        "You are given a claim and a connection options for the entities present in the claim. "
        "Your task is to filter the Connections options that could be relevant for the claim"
    )

    content = f'''
    ## TASK:
    For each of the given entities given in the DICT structure below as evidence: 
    Filter the connections strictly from the given options that would be relevant to connect given entities to fact-check the given claim.
    Think clever, there could be multi-step hidden connections, if not direct, that could connect the entities somehow.

    Claim: {claim}
    Evidence as DICT structure: {evidence}

    ### Answer Format:
    Return the relavant entities for fact-checking the evidence as DICT structure.
    No code output. No explanation. Output only valid python DICT of structure like the one given to you.
    '''
   
    return instruction, content


# !!! THESE TOKENS SHOULD BE KEPT PRIVATE AND NOT SHARED !!!
cloudfare_api = "Bearer UNdagTbhR_O3WkU0qhVfO22vmVleKw4YWFy2Gbik"
account_id = "307ed229b30d79addccada26fcf4bc7d"



API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
headers = {"Authorization": cloudfare_api}

def run(model, inputs):
    input = { "messages": inputs }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

def send_request(instruction, content):
    # print("INSTRUCTION AND CONTENT")
    # print(instruction)
    # print(content)
    inputs = [
        { "role": "system", "content": instruction},
        { "role": "user", "content": content}
    ];
    output = run("@cf/meta/llama-3.2-3b-instruct", inputs)
    print(output)
    response = output['result']['response']
    return response


with open('with_evidence/prompting/devset_small.pickle', 'rb') as file:
    # there is this 946th claim that contains some harmful information that llm refuse to answer
    data = pickle.load(file)


for claim, information in data.items():
    evidence = information['evidence_onehop_full']
    # print("Original:", evidence)
    instruction, content = shorten_evidence(claim, evidence)
    print(len(evidence))
    response = send_request(instruction, content)
    print(len(response))
    # print("SHORTENED:", response)

