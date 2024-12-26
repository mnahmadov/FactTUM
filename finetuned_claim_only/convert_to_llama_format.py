import pickle
import csv
import pandas as pd


# train_file_path = "claim_only/factkg_dataset/factkg_train.pickle"
# instruction = f'''You are an advanced AI fact-checker trained on a vast corpus, including information from Wikipedia. \
# You are given a claim, and your task is to verify whether all the facts in the claim are supported by \
# your knowledge base. Use your understanding of the world and factual data to make an accurate judgment. \
# Your response must follow the given format exactly as specified below, without any deviation.
# ### Response Format: Your response should be a single word answer "True" or "False
        
# ## TASK:
# Verify the given input claim.

# ### Instructions:
# Assess whether the claim is true or false based on your knowledge.

# ### Answer Format:
# Provide your response as a single word answer "True" or "False"'''

# def write_to_csv(filename, output, input, instruction):
#     fieldnames = ["output", "input", "instruction"]
    
#     with open(filename, mode="a", newline="", encoding="utf-8") as file:
#         writer = csv.DictWriter(file, fieldnames=fieldnames)
        
#         # If the file is empty, write the header
#         if file.tell() == 0:
#             writer.writeheader()
        
#         writer.writerow({"output": output, "input": input, "instruction": instruction})

# with open(train_file_path, 'rb') as file:
#     data = pickle.load(file)


# for claim, information in data.items():
#     input = claim
#     output = information['Label'][0]
#     write_to_csv("finetuned_claim_only/train_llama_format.csv", output, input, instruction)


df = pd.read_csv("finetuned_claim_only/train_llama_format.csv")

print(df['instruction'])
