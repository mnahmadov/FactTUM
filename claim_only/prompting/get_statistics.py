import pandas as pd
from sklearn.metrics import accuracy_score

EVALUATION_FILE_PATH = "claim_only/prompting/prompting_claim_only_v0.csv"
# Read the CSV file containing true and predicted labels
df = pd.read_csv(EVALUATION_FILE_PATH, dtype={'true_label': bool, 'predicted_label': bool})

# Overall accuracy
accuracy = accuracy_score(df['true_label'], df['predicted_label'])

reasoning_type_accuracies = {'total': accuracy}

# Group the dataframe by 'reasoning_type' and calculate accuracy for each group
for reasoning_type, group in df.groupby('reasoning_type'):
    accuracy = accuracy_score(group['true_label'], group['predicted_label'])
    reasoning_type_accuracies[reasoning_type] = accuracy

print(reasoning_type_accuracies)