import sys
from apetokenizer.src.apetokenizer.ape_tokenizer import APETokenizer
import pandas as pd
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from backbone.backbone import ModelXtoCtoY_function
import numpy as np
from sklearn.metrics import roc_auc_score
from utils import set_seed, agent
import ast
import pickle as pkl
from utils import MyDataset
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('args.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

set_seed(config['seed'])
data_type = config['data_type']
num_epochs = config['num_epochs']
num_concepts = config['num_concepts']

tokenizer = APETokenizer()
tokenizer.load_vocabulary('apetokenizer/tokenizer.json')
model = AutoModelForSequenceClassification.from_pretrained('mikemayuare/SMILY-APE-BBBP').to(device)

# load data
DATA = {}
DATA['train'] = pd.read_csv(f'data/train_{data_type}.csv')
DATA['val'] = pd.read_csv(f'data/val_{data_type}.csv')
DATA['test'] = pd.read_csv(f'data/test_{data_type}.csv')

# choose num_concepts features with llm agent
features = agent(data_type, DATA['train'].drop(columns=['Drug', 'Y', 'Drug_ID']).columns.tolist(), num_concepts)
features = ast.literal_eval(features)

with open(f'explanations_dir/features_llm_{data_type}.pkl', 'wb') as f:
    pkl.dump(features, f)

# means and std for standardization
means = DATA['train'][features].mean().values
stds = DATA['train'][features].std().values

# create the dataloader
train_loader = DataLoader(MyDataset('train', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=True)
val_loader = DataLoader(MyDataset('val', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=False)
test_loader = DataLoader(MyDataset('test', features, means, stds, tokenizer, DATA), batch_size=8, shuffle=False)

# num_concepts is the number of concepts, expand_dim is the dimension of the expanded layer (0 means no expansion)
ModelXtoCtoY_layer = ModelXtoCtoY_function(num_concepts=num_concepts, expand_dim=0, in_dims=in_dims).to(device)

loss_C = torch.nn.L1Loss()
loss_Y = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(list(ModelXtoCtoY_layer.parameters()), lr=1e-5)

best_acc_score = 0
for epoch in range(num_epochs):
    ######### train #########
    ModelXtoCtoY_layer.train()
    model.eval()

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        concept_labels = batch['concept_labels']

        optimizer.zero_grad()
        with torch.no_grad():
            outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0]

        outputs = ModelXtoCtoY_layer(pooled_output)

        XtoC_output = outputs[1:] 
        XtoY_output = outputs[0:1]

        # XtoC_loss
        XtoC_output = torch.stack(XtoC_output, dim=1).squeeze()
        XtoC_loss = loss_C(XtoC_output.to(device), concept_labels.squeeze().to(device))
    
        # XtoY_loss
        XtoY_loss = loss_Y(XtoY_output[0].squeeze().to(device), label.squeeze().to(device))

        loss = XtoY_loss + XtoC_loss * float(sys.argv[8])
        
        loss.backward()
        optimizer.step()

    ######### val #########
    ModelXtoCtoY_layer.eval()
    model.eval()
    val_accuracy = 0.
    concept_val_loss = 0.
    predict_labels = np.array([])

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['label'].to(device)
            concept_labels = batch['concept_labels']

            outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
            pooled_output = outputs.hidden_states[-1][:,0]

            outputs = ModelXtoCtoY_layer(pooled_output)
            XtoY_output = outputs[0:1]

            predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().cpu() > 0.5) == label.bool().cpu())

        val_accuracy = predict_labels.sum() / len(predict_labels)
        
    if val_accuracy > best_acc_score:
        best_acc_score = val_accuracy
        torch.save(model, f'explanations_dir/model_llm_{data_type}.pth')
        torch.save(ModelXtoCtoY_layer, f'explanations_dir/ModelXtoCtoY_layer_llm_{data_type}.pth')

######### test #########
model = torch.load(f'explanations_dir/model_llm_{data_type}.pth', weights_only=False)
ModelXtoCtoY_layer = torch.load(f'explanations_dir/ModelXtoCtoY_layer_llm_{data_type}.pth', weights_only=False) 
model.eval()
ModelXtoCtoY_layer.eval()

predict_labels = np.array([])
true_labels = np.array([])
predictions = np.array([])
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        concept_labels = batch['concept_labels']

        outputs = model(input_ids=input_ids.squeeze(), attention_mask=attention_mask.squeeze(), output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1][:,0]

        outputs = ModelXtoCtoY_layer(pooled_output)
        XtoY_output = outputs[0:1]
        predictions = np.append(predictions, XtoY_output[0].squeeze().to(torch.float32).cpu())
        predict_labels = np.append(predict_labels, (XtoY_output[0].squeeze().to(torch.float32).cpu() > 0.0) == label.bool().cpu())

        true_labels = np.append(true_labels, label.bool().cpu())

    test_accuracy = predict_labels.sum() / len(predict_labels)

    with open(f'explanations_dir/test_loader_llm_{data_type}.pkl', 'wb') as f:
        pkl.dump(test_loader, f)
    print(f'Test Acc = {test_accuracy*100}')
    print(f'Test roc_auc_score = {roc_auc_score(true_labels, predictions)}')