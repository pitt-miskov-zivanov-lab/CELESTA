__author__ = 'difei'

import torch
import argparse
import os
import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging

from semilearn import get_net_builder, BasicDataset
from semilearn.datasets.collactors import get_biobert_collactor
from preprocessing.labels_loader import *

# load utils
import utils

model_name = "dmis-lab/biobert-v1.1"
utils.tokenizer = AutoTokenizer.from_pretrained(model_name)
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from msp import UlbDataset, collate_fn

def msp_filter(text_list, task=None):
    '''
    Args:
        input_path: the path of the msp model
        text_list: the list of text to be detected
        task: the task name
    '''
    # msp
    input_path = './saved_models/base/'
    model_path = os.path.join(input_path, f'{task}.pth')

    train_dataset = UlbDataset(text_list, [0]*len(text_list))
    dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=False, batch_size=1)

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    for step, (data, label, indices) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            data.to(device)
            outputs = model(**data)

        logits = outputs.logits
        confidences = torch.softmax(logits, dim=1).max(dim=-1).values

        # threshold
        outlier_indices = (confidences < 0.9).nonzero().squeeze()
        print(confidences)
    
        if outlier_indices.numel() == 0:
            return False 
        else:
            # OOD sample is detected 
            return True

def semi_inference(input_path, text_list, task=None, alg=None):
    # semi-learn
    model_path = os.path.join(input_path, f'{task}/model_best.pth')

    checkpoint = torch.load(model_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item

    model = get_net_builder("biobert", from_name=False)(num_classes=num_classes[context_types.index(task)])
    keys = model.load_state_dict(load_state_dict)

    dataset = BasicDataset(alg, text_list, [0] * len(text_list), num_classes[context_types.index(task)],
                           is_ulb=False)

    collact_fn = get_biobert_collactor(max_length=512)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collact_fn, drop_last=False, shuffle=False)

    model.eval()

    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for data in data_loader:
            x = data['x_lb']
            y = data['y_lb']

            logits = model(x)['logits']

            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_logits = np.concatenate(y_logits)

    # map id to label
    id = y_pred[0]
    id2label = context_mapping[task][0]
    label = id2label[id]

    print(label)


def mtl_inference(input_path, text_list, task=None, alg=None):
    # semi-learn
    model_path = os.path.join(input_path, f'{task}/model_best.pth')

    checkpoint = torch.load(model_path)
    load_model = checkpoint['model']
    # print("load---", load_model.keys())

    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item

    tasks_info = [task, 'relation']
    num_classes = [len(task_classes[task]), len(RELATION)]

    # print('\n\n')

    model = get_net_builder("biobert", from_name=False)(num_classes=num_classes, use_mtl=True)
    keys = model.load_state_dict(load_state_dict)

    dataset = BasicDataset(alg, text_list, [0] * len(text_list), num_classes, is_ulb=False)
    collact_fn = get_biobert_collactor(max_length=512)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collact_fn, drop_last=False, shuffle=False)

    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for data in data_loader:
            x = data['x_lb']
            y = data['y_lb']

            logits = model(x)['logits']
            logits = logits[tasks_info.index(task)]

            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_logits = np.concatenate(y_logits)

    # map id to label
    id = y_pred[0]
    id2label = context_mapping[task][0]
    label = id2label[id]

    print(label)

# inference
def test():
    input_path="./saved_models/mtl+vat"
    task = 'cell_type'
    alg = 'mtl_vat'

    text_list = ['This is a test sentence.']
    if msp_filter(text_list, task):
        print('outlier is founded')
    else:
        mtl_inference(input_path, text_list, task, alg)
