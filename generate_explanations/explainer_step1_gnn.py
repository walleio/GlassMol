import torch
import sys
import pickle as pkl
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy.ma as ma
from utils import MolNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = sys.argv[1] # whichever dataset you have trained

# load the model
model = torch.load(f'explanations_dir/model_gnn_{dataset}.pth', weights_only=False).to(device)
ModelXtoCtoY_layer = torch.load(f'explanations_dir/ModelXtoCtoY_layer_gnn_{dataset}.pth', weights_only=False).to(device)
model.eval()
ModelXtoCtoY_layer.eval()

# load the saved test loader
with open(f'explanations_dir/test_loader_gnn_{dataset}.pkl', 'rb') as f:
           test_loader = pkl.load(f)

# load the features
with open(f'explanations_dir/features_gnn_{dataset}.pkl', 'rb') as f:
    features = pkl.load(f)

# get the contributions for the entire test set
contributions_dict = {}
all_contributions = []
for batch in test_loader:
    label = batch.y

    outputs = model(batch.to(device))

    outputs = ModelXtoCtoY_layer(outputs)
    concepts = torch.stack(outputs[1:], dim=1)

    last_layer = None
    for name, m in ModelXtoCtoY_layer.named_modules():
        if name == 'sec_model':
            last_layer = m

    for name, m in last_layer.named_modules():
        if name == 'linear':
            last_layer = m

    # get the weights for the last layer
    W = last_layer.weight.squeeze()

    # contribution calculation as seen in the paper
    contributions = concepts.squeeze().detach().cpu().numpy()*W.detach().cpu().numpy()
    all_contributions.append(contributions.copy())

    for i, feature in enumerate(features):
        try:
            contributions_dict[feature] = np.append(contributions_dict[feature], ((contributions.T)[i]))
        except:
            contributions_dict[feature] = ((contributions.T)[i])       

with open (f'explanations_dir/contributions_gnn_{dataset}.pkl', 'wb') as f:
    pkl.dump(contributions_dict, f)
    print('\a')