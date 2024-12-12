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

def semi_evaluate(input_path, output_path=None, task=None, alg=None, batch_size=None):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    logging.basicConfig(filename=os.path.join(output_path, f"{task}_log.txt"), filemode="w",
                        format="%(asctime)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

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

    test_df = get_test_df(task=task)
    test_lb_text_list, test_lb_label_list, _, _ = split_ssl_list(test_df, task=task)
    eval_dataset = BasicDataset(alg, test_lb_text_list, test_lb_label_list, num_classes[context_types.index(task)],
                                is_ulb=False)

    # TODO: create loader in semi-learn settings
    collact_fn = get_biobert_collactor(max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collact_fn, drop_last=False, shuffle=False)

    model.eval()

    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for data in eval_loader:
            x = data['x_lb']
            y = data['y_lb']

            logits = model(x)['logits']

            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.append(torch.softmax(logits, dim=-1).cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_logits = np.concatenate(y_logits)

    p, r, f, s = precision_recall_fscore_support(y_pred=y_pred, y_true=y_true, average="micro")
    logger.info("precision: {:.4f}".format(p))
    logger.info("recall: {:.4f}".format(r))
    logger.info("f1: {:.4f}".format(f))

    # save dataframe for error analysis
    pred_output = copy.deepcopy(test_df)
    pred_output[f'{task}_pred'] = y_pred

    map_ids2label(pred_output, task)

    pred_output.style.apply(custom_style, columns=[task, f'{task}_pred'], axis=1).to_excel(os.path.join(output_path, f'{task}_output.xlsx'))

def mtl_evaluate(input_path, output_path=None, task=None, alg=None, batch_size=None):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    logging.basicConfig(filename=os.path.join(output_path, f"{task}_log.txt"), filemode="w",
                        format="%(asctime)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    
    # semi-learn
    model_path = os.path.join(input_path, f'{task}/model_best.pth')
    
    checkpoint = torch.load(model_path)

    load_model = checkpoint['model'] # FIXME
    #print("load---", load_model.keys())
    
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item

    tasks_info = [task, 'relation']
    num_classes = [len(task_classes[task]), len(RELATION)]

    #print('\n\n')
    
    model = get_net_builder("biobert", from_name=False)(num_classes=num_classes, use_mtl=True)
    keys = model.load_state_dict(load_state_dict)
    #print("model---", model.state_dict().keys())   

    model.eval()

    # only evaluate context, ignore relation
    test_df_task = get_test_df(task=task)
    test_df_task.dropna(inplace=True)
    test_lb_text_list, test_lb_label_list, _, _ = split_ssl_list(test_df_task, task=task)
    eval_dataset = BasicDataset(alg, test_lb_text_list, test_lb_label_list, num_classes, is_ulb=False)
    collact_fn = get_biobert_collactor(max_length=512)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collact_fn, drop_last=False, shuffle=False)

    y_true = []
    y_pred = []
    y_logits = []
    with torch.no_grad():
        for data in eval_loader:
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

    p, r, f, s = precision_recall_fscore_support(y_pred=y_pred, y_true=y_true, average="micro")
    logger.info(f"{task}:")
    logger.info("precision: {:.4f}".format(p))
    logger.info("precision: {:.4f}".format(p))
    logger.info("recall: {:.4f}".format(r))
    logger.info("f1: {:.4f}".format(f))

    # save dataframe for error analysis
    pred_output = copy.deepcopy(test_df_task)
    pred_output[f'{task}_pred'] = y_pred

    map_ids2label(pred_output, task)

    pred_output.style.apply(custom_style, columns=[task, f'{task}_pred'], axis=1).to_excel(os.path.join(output_path, f'{task}_output.xlsx'))


ap = argparse.ArgumentParser()
ap.add_argument("--input", default='./saved_models/vat20/', required=False, type=str,
                help="input file path for model checkpoints")
ap.add_argument("--task", default='cell_line', required=False, type=str, help="Task name")
ap.add_argument("--alg", default='vat', required=False, type=str, help="SSL algorithm name")
ap.add_argument("--output", default='./evaluated_results/vat20/', required=False, type=str,
                help="output path")
ap.add_argument("--batch", default=8, required=False, type=int, help="batch size")
ap.add_argument("--eval_mtl", default=False, required=False, type=bool,
                help="evaluate semi-learn or in muti-task-learn settings")
args = ap.parse_args()

input_path = args.input
task = args.task
alg = args.alg
output_path = args.output
batch_size = args.batch
eval_mtl = args.eval_mtl

if eval_mtl:
    mtl_evaluate(input_path, output_path=output_path, task=task, alg=alg, batch_size=batch_size)
else:
    semi_evaluate(input_path, output_path=output_path, task=task, alg=alg, batch_size=batch_size)