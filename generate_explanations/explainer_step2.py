import numpy as np
import pickle as pkl
import pandas as pd
import sys

model_type = sys.argv[1] # 'gnn' or 'llm'
dataset = sys.argv[2] # whichever dataset you have trained and run step 1 on

analyze_mol = 42 # index of the molecule in the test set to analyze

with open(f'explanations_dir/contributions_{model_type}_{dataset}.pkl', 'rb') as f:
    contributions = pkl.load(f)

data = pd.read_csv(f'data/test_{dataset}.csv').iloc[analyze_mol]

# create a dictionary of the contributions for the molecule
contributions_2 = {data['Drug_ID']: []}

for k, v in contributions.items():
    for i in range(len(v)):
        if i == analyze_mol:
            contributions_2[data['Drug_ID']].append({'name': k, 'value': v[i]})

# sort the contributions by absolute value
for k, v in contributions_2.items():
    contributions_2[k] = sorted(v, key=lambda x: abs(x['value']), reverse=True)

# retain the features that have non-zero values in the original data
contributions_3 = {}
for k, v in contributions_2.items():
    contributions_3[k] = [i for i in v if data[i['name']].item() != 0]

print(f'{data["Drug_ID"]}:')
for idx, i in enumerate(contributions_3[data['Drug_ID']]):
    print(f'{i["name"]}: {i["value"]:.4f}')
    print('-'*(len(str(f'{i["name"]}: {i["value"]:.4f}'))-0))
    if idx > 1:
        print('...')
        break