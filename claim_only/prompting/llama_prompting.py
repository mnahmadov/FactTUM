import requests
import pickle
import ast
import csv
import pandas as pd


def claim_only_llm(claim):
    # This function is to prepare the instruction and content for the LLM

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


# Cloudflare API token and account information (Keep private)
cloudfare_api = "Bearer cloudfare_api" # exactly replace cloudfare_api with actual api, keeping Bearer and space before it
account_id = "account_id"

# Prepare the API base URL for requests
API_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
headers = {"Authorization": cloudfare_api}

def run(model, inputs):
    # This function is to make requests
    input = { "messages": inputs }
    response = requests.post(f"{API_BASE_URL}{model}", headers=headers, json=input)
    return response.json()

def send_request(instruction, content):
    # This function is to prepare input for the LLM and to call the API
    inputs = [
        { "role": "system", "content": instruction},
        { "role": "user", "content": content}
    ];
    output = run("@cf/meta/llama-3.2-3b-instruct", inputs)
    response = output['result']['response']
    return response

def str_to_lst(response):
    # This function is to convert the string response (list string) to a list format
    try:
        response = ast.literal_eval(response)
        # Check if the string is in a list format and it has 3 entries inside
        if not isinstance(response, list) or len(response) != 3:
            return False
        if isinstance(response[0], str): 
            # Convert the true/false strings to boolean values
            stripped = response[0].strip().lower()
            if stripped == "true":
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
    # This function is to extract reasoning type for the given claim
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
    # This function is to write the results to a CSV file
    fieldnames = ["claim", "reasoning_type", "true_label", "predicted_label"]
    
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # If the file is empty, write the header
        if file.tell() == 0:
            writer.writeheader()
        
        writer.writerow({"claim": claim, "reasoning_type": reasoning_type, "true_label": true_label, "predicted_label": predicted_label})

# Load the factKG test set
with open('factkg_dataset/factkg_test.pickle', 'rb') as file:
    data = pickle.load(file)

i = 0
for claim, information in data.items():
    # if i < 8930: # uncomment if a particular claim cannot be verified by the model e.g. harmful information
    #     i += 1
    #     continue
    instruction, content = claim_only_llm(claim) # generate instruction and content for the LLM
    response = send_request(instruction, content) # send request and get LLM response


    response = str_to_lst(response) # Convert the response into a valid list format
    while not response: # Retry if the response format is invalid
        response = send_request(instruction, content)
        response = str_to_lst(response)

    # Get the necessary properties and write the results to CSV file
    true_label = information['Label'][0]
    predicted_label = response[0]
    reasoning_type = get_reasoning_type(claim, data)[0]
    PATH_TO_RESULT_FILE = "claim_only/prompting/prompting_claim_only_new.csv"
    write_to_csv(PATH_TO_RESULT_FILE, claim, reasoning_type, true_label, predicted_label)
