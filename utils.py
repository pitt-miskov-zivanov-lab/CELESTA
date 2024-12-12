__author__ = 'difei'

import copy
import os
import pandas as pd
import numpy as np
from preprocessing.labels_loader import context_mapping
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from preprocessing.labels_loader import aug_cols, context_types, task_classes

# default version
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_embeddings(embeddings, labels, output_path, task):
    # Apply t-SNE and set perplexity for testing
    perplexity = min(30, len(embeddings) - 1) if len(embeddings) > 1 else 1  

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plotting the t-SNE reduced embeddings
    #plt.figure(figsize=(10, 8), dpi=600)
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='jet', alpha=0.7)
    
    # set color bar
    cbar = plt.colorbar(scatter, ticks=range(min(labels), max(labels) + 1))
    cbar.set_label('Labels')
    cbar.set_ticks(range(min(labels), max(labels) + 1))
    cbar.set_ticklabels([task_classes[task][i] for i in range(min(labels), max(labels) + 1)])

    plt.title(f't-SNE Visualization of {task} Embeddings')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')

    # Save the plot
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    plt.savefig(os.path.join(output_path, f'{task}_tsne_plot.png'))
    plt.show()

def custom_style(row, columns):
    true_col = columns[0]
    pred_col = columns[1]
    if row[true_col] != row[pred_col]:
        return ['background-color: yellow'] * len(row)
    else:
        return [''] * len(row)

def map_to_int(x):
    try:
        return int(x)
    except ValueError:
        return np.nan

def map_label2ids(df: pd.DataFrame, col_name: str):
    label2id = context_mapping[col_name][1]
    label2id = copy.deepcopy(label2id)
    if 'ood' in df.values:
        label2id['ood'] = -100

    df[col_name] = df[col_name].map(label2id, na_action='ignore')
    df[col_name] = df[col_name].map(map_to_int)

def map_ids2label(df: pd.DataFrame, col_name: str):
    id2label = context_mapping[col_name][0]
    id2label = copy.deepcopy(id2label)

    df[col_name] = df[col_name].map(id2label, na_action='ignore')
    df[f'{col_name}_pred'] = df[f'{col_name}_pred'].map(id2label, na_action='ignore')

def collate_fn(exmaples):
    return tokenizer.pad(exmaples, padding="longest", return_tensors="pt")

def preprocess_function(examples):
    outputs = tokenizer(examples["text"], padding=True, truncation=True, max_length=512)
    return outputs

def get_test_df(test_path=None, task=None):
    if not test_path:
        test_path = "./data/large_context_corpus_test.xlsx"
    
    test_df = pd.read_excel(test_path)[['text', task]]

    map_label2ids(test_df, task)
    test_df.dropna(inplace=True)
    return test_df

def get_dataset(df: pd.DataFrame, task=None):
    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column(task, "labels")
    dataset = dataset.with_format("torch", columns=["labels"], dtype=int)

    remove_columns = dataset.column_names
    remove_columns.remove("labels")
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=remove_columns)
    return tokenized_dataset

def get_dataloader(tokenized_dataset: Dataset, is_eval=True, batch_size=None):
    shuffle = False
    if not is_eval:
        shuffle = True

    if not batch_size:
        batch_size = 8

    return DataLoader(tokenized_dataset, shuffle=shuffle, collate_fn=collate_fn, batch_size=batch_size)

def split_ssl_list(df: pd.DataFrame, task=None, alg=None):
    df_cols = df.columns.tolist()
    assert 'text' in df_cols, "text column not found in the input"
    
    label_list = df[task].to_list()
    text_list = df['text'].to_list()

    ood_v = float(-100)
    ulb_v = float(-200)

    # filter out OOD data  
    # augmented data as input
    if alg == 'fixmatch':
        assert 'umls' in df_cols, "umls column not included in the input"
        for col in aug_cols:
            assert col in df_cols, f"{col} column not included in the input"

        text_aug1_list = df['umls'].to_list()
        text_aug2_list = df['sr'].to_list()
        text_aug3_list = df['ri'].to_list()

        ulb_text_list = [(text_aug1_list[idx], text_aug2_list[idx], text_aug3_list[idx]) for idx, x in enumerate(label_list) if x == ulb_v]

        # TODO: comatch, remixmatch
        # ulb_text_list = [(text_list[idx], text_aug1_list[idx], text_aug2_list[idx]) for idx, x in enumerate(label_list) if x == 'none' and x != 'ood']
    else: 
        ulb_text_list = [(text_list[idx], 'None', 'None') for idx, x in enumerate(label_list) if x == ulb_v]

    lb_text_list = [(text_list[idx], 'None', 'None') for idx, x in enumerate(label_list) if x != ulb_v and x != ood_v]
    
    ulb_label_list = [x for x in label_list if x == ulb_v]
    lb_label_list = [x for x in label_list if x != ulb_v and x != ood_v]

    return np.array(lb_text_list), np.array(lb_label_list).astype(int), np.array(ulb_text_list), np.array(ulb_label_list)

def split_mtl_ssl_list(df, task=None, alg=None):
    df_cols = df.columns.tolist()

    text_list = df['text'].to_list()
    task_label_list = df[task].to_list() # e.g., location, cell_line..
    relation_label_list = df['relation'].to_list()

    ood_v = float(-100)
    ulb_v = float(-200)

    ulb_idx = [idx for idx, x in enumerate(task_label_list) if x == ulb_v]
    lb_idx = [idx for idx, x in enumerate(task_label_list) if x != ulb_v and x != ood_v]

    if alg == 'mtl_fixmatch':
        assert 'umls' in df_cols, "umls column not included in the input"
        for col in aug_cols:
            assert col in df_cols, f"{col} column not included in the input"

        text_aug1_list = df['umls'].to_list()
        text_aug2_list = df['sr'].to_list()
        text_aug3_list = df['ri'].to_list()

        ulb_text_list = [(text_list[idx], text_aug1_list[idx], text_aug1_list[idx]) for idx in ulb_idx]
    else:
        ulb_text_list = [(text_list[idx], 'None', 'None') for idx in ulb_idx]

    lb_text_list = [(text_list[idx], 'None', 'None') for idx in lb_idx]

    relation_ulb_label_arr = np.array([relation_label_list[idx] for idx in ulb_idx])
    task_ulb_label_arr = np.array([task_label_list[idx] for idx in ulb_idx])
    relation_lb_label_arr = np.array([relation_label_list[idx] for idx in lb_idx])
    task_lb_label_arr = np.array([task_label_list[idx] for idx in lb_idx])

    ulb_label_arr = np.column_stack((task_ulb_label_arr, relation_ulb_label_arr))
    lb_label_arr = np.column_stack((task_lb_label_arr, relation_lb_label_arr))

    return np.array(lb_text_list), lb_label_arr.astype(int), np.array(ulb_text_list), ulb_label_arr.astype(int)

