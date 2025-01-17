import requests
import pickle
import ast
import csv
import pandas as pd


def claim_only_llm(claim):
    instruction = (
        "You are an advanced AI fact-checker trained on a vast corpus, including information from Wikipedia. "
        "You are given a claim, and your task is to verify whether all the facts in the claim are supported by "
        "your knowledge base. Use your understanding of the world and factual data to make an accurate judgment. "
        "Your response must follow the given format exactly as specified below, without any deviation. "
        "### Response Format:"
        "Your response should be a list with the following structure and have nothing else than this list structure: "
        "[True/False, 'one sentence of your decision', 'sources used'] "
        "Ensure that you answer with only the three parts specified, in the correct order."
    )

    content = f'''
    ## TASK:
    Verify the following claim:

    Claim: {claim}

    ### Instructions:
    Assess whether the claim is true or false based on your knowledge.

    ### Answer Format:
    Provide your response as a list with the following items:
    1. "True" or "False" (single word answer)
    2. One-sentence explanation of your decision
    3. Sources you used, if any (link or reference)

    ### Example 1:
    Claim: "The Earth is flat."
    Response: ["False", "The Earth is round as confirmed by extensive scientific research and satellite imagery.", "NASA, Scientific American"]

    ### Example 2:
    Claim: "Albert Einstein developed the theory of relativity."
    Response: ["True", "Albert Einstein is widely recognized for formulating the theory of relativity.", "Einstein's published papers, Nobel Prize archives"]
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

def str_to_lst(response):
    try:
        response = ast.literal_eval(response)
        if not isinstance(response, list) or len(response) != 3:
            return False
        # print(response)
        # print(type(response[0]))
        if isinstance(response[0], str):
            stripped = response[0].strip().lower()
            # print(stripped)
            if stripped == "true": # here i convert the true/false strings to boolean values
                response[0] = True
                return response
            elif stripped == "false":
                response[0] = False
                return response
        else:
            return False
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing response string: {e}")
        return False

def get_reasoning_type(claim, test_data):
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
    fieldnames = ["claim", "reasoning_type", "true_label", "predicted_label"]
    
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writeheader()
        
        writer.writerow({"claim": claim, "reasoning_type": reasoning_type, "true_label": true_label, "predicted_label": predicted_label})


with open('claim_only/factkg_dataset/factkg_test.pickle', 'rb') as file:
    # there is this 946th claim that contains some harmful information that llm refuse to answer
    data = pickle.load(file)

print(len(data.items()))
i = 0
for claim, information in data.items():
    if i < 8930: # this is to avoid the 946th claim that is harmful
        i += 1
        continue
    instruction, content = claim_only_llm(claim)
    response = send_request(instruction, content)
#     print(response)
#     print("Claim: ", claim)
#     print("Original Response: ",response)

    response = str_to_lst(response) # try to convert the response into an actual list
    while not response: # if not in the correct format
        response = send_request(instruction, content)
        response = str_to_lst(response)

    true_label = information['Label'][0]
    predicted_label = response[0]
    reasoning_type = get_reasoning_type(claim, data)[0]
    write_to_csv("claim_only/llama_prompting/prompting_claim_only_v2.csv", claim, reasoning_type, true_label, predicted_label)
