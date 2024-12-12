import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import copy
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from utils import tokenizer, map_label2ids
from preprocessing.labels_loader import context_types
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_path = os.path.join('./data/', 'large_context_corpus_train.xlsx')

class UlbDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        samp = {"text": self.data[index], "labels": self.labels[index]}
        results = tokenize_function(samp)
        results['labels'] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        results['idx'] = torch.tensor(int(index), dtype=torch.long)
        return results

    def __len__(self):
        return len(self.data)

def collate_fn(features):
    w_features = []
    batch_label = []
    batch_indices = []
    for f in features:
        f_ = {k:v for k,v in f.items() if 'labels' not in k and 'idx' not in k}
        w_features.append(f_)
        batch_label.append(f['labels'])
        batch_indices.append(f['idx'])
    padded_texts = tokenizer.pad(w_features, padding="longest", return_tensors="pt")
    batch_label = torch.tensor(batch_label, dtype=torch.long)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    return padded_texts, batch_label, batch_indices

def tokenize_function(example):
    outputs = tokenizer(example["text"], padding=True, truncation=True, max_length=512)
    return outputs

def main(input_path, output_path=None, task=None, batch_size=None):
    # get train dataset
    train_df = pd.read_excel(train_path)

    if os.path.isdir(input_path) and not task:
        for task in context_types:
            # FIXME: species
            model_path = os.path.join(input_path, f'{task}.pth')
            train_df = msp(model_path=model_path, train_df=train_df, task=task, batch_size=batch_size)
    else:
        assert task != None, "task should be explicitly defined"
        # msp for individual task
        model_path = input_path
        train_df = msp(model_path=model_path, train_df=train_df, task=task, batch_size=batch_size)

    train_df.to_excel(os.path.join(output_path, f'large_context_corpus_train_ood.xlsx'))

def msp(model_path, train_df, task=None, batch_size=None):
    task_df = pd.read_excel(train_path)[['text', task]]
    map_label2ids(task_df, task)
    task_df.fillna(-100, inplace=True)

    label_list = task_df[task].to_list()
    text_list = task_df['text'].to_list()
    train_dataset = UlbDataset(text_list, label_list)

    dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=False, batch_size=batch_size)

    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    ood_indices = []
    id_indices = []
    for step, (data, label, indices) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            data.to(device)
            outputs = model(**data)

        logits = outputs.logits
        prediction = logits.argmax(dim=-1)

        logits = outputs.logits

        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = torch.softmax(logits, dim=1).max(dim=-1).values

        #print(predictions, print(confidences))

        # threshold
        outlier_indices = (confidences < 0.9).nonzero().squeeze()

        # filtering
        # make sure unlabeled data are notated as -100
        if outlier_indices.dim() != 0:
            for i in outlier_indices:
                if label.tolist()[i] == -100:
                    ood_idx = indices.tolist()[i]
                    ood_indices.append(ood_idx)

    train_df.iloc[ood_indices, train_df.columns.get_loc(task)] = 'ood'
    return train_df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default='./saved_models/base/', required=True, type=str,
                    help="saved model checkpoints")
    ap.add_argument("--task", default=None, required=False, type=str, help="Task name")
    ap.add_argument("--output", default='./data/', required=False, type=str, help="output path")
    ap.add_argument("--batch", default=8, required=False, type=int, help="batch size")
    args = ap.parse_args()

    model_path = args.input
    task = args.task
    output_path = args.output
    batch_size = args.batch

    main(model_path, output_path, task=task, batch_size=batch_size)