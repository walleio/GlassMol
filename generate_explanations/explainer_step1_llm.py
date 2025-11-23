import torch
import sys
import pickle as pkl
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = sys.argv[1] # whichever dataset you have trained

# load the model
model = torch.load(f'explanations_dir/model_llm_{dataset}.pth', weights_only=False)
ModelXtoCtoY_layer = torch.load(f'explanations_dir/ModelXtoCtoY_layer_llm_{dataset}.pth', weights_only=False)
ModelXtoCtoY_layer.eval()
model.eval()

# load the saved test loader
with open(f'explanations_dir/test_loader_llm_{dataset}.pkl', 'rb') as f:
    test_loader = pkl.load(f)

# load the features
with open(f'explanations_dir/features_llm_{dataset}.pkl', 'rb') as f:
    features = pkl.load(f)

# get the contributions for the entire test set
all_contributions = []
contributions_dict = {}
for batch in test_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']
    concept_labels = batch['concept_labels']
    features = batch['features']
            
    outputs = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), output_hidden_states=True)

    pooled_output = outputs.hidden_states[-1][:,0] 

    outputs = ModelXtoCtoY_layer(pooled_output)
    concepts = torch.stack(outputs[1:], dim=1)
          
    last_layer = None
    for name, m in ModelXtoCtoY_layer.named_modules():
        if name == 'sec_model':
            last_layer = m

    for name, m in last_layer.named_modules():
        if name == 'linear':
            last_layer = m

    W = last_layer.weight.squeeze()

    # contribution calculation as seen in the paper
    contributions = concepts.squeeze().detach().cpu().numpy()*W.detach().cpu().numpy()
    all_contributions.append(contributions.copy())

    for i, feature in enumerate(features):
        try:
            contributions_dict[feature[0]] = np.append(contributions_dict[feature[0]], ((contributions.T)[i]))
        except:
            contributions_dict[feature[0]] = ((contributions.T)[i])

with open (f'explanations_dir/contributions_llm_{dataset}.pkl', 'wb') as f:
    pkl.dump(contributions_dict, f)
    print('\a')