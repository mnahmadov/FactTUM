import pandas as pd
from sklearn.metrics import accuracy_score


df = pd.read_csv('with_evidence/prompting/prompting_with_evidence_v0.csv', dtype={'true_label': bool, 'predicted_label': bool})

accuracy = accuracy_score(df['true_label'], df['predicted_label'])

reasoning_type_accuracies = {'total': accuracy}
for reasoning_type, group in df.groupby('reasoning_type'):
    accuracy = accuracy_score(group['true_label'], group['predicted_label'])
    reasoning_type_accuracies[reasoning_type] = accuracy

print(reasoning_type_accuracies)

