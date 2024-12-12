import pandas as pd
import torch
import os
from transformers import AutoTokenizer
import numpy as np
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForSequenceClassification

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support
from transformers import EvalPrediction

from transformers import TrainingArguments, Trainer
from preprocessing.labels_loader import *
from utils import map_label2ids, get_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = "dmis-lab/biobert-v1.1"
metric_name = "f1"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    outputs = tokenizer(examples["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")
    return outputs

def dataset(train_path, test_path, task):

    # read only subset of data - text and task columns here
    train_df = pd.read_excel(train_path)[['text', task]]
    test_df = pd.read_excel(test_path)[['text', task]]

    # remove nan
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    map_label2ids(train_df, task)
    map_label2ids(test_df, task)

    train_dataset = get_dataset(train_df, task=task)
    test_dataset = get_dataset(test_df, task=task)

    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    metrics = {'f1': f1,
            'precisioin': precision,
            'recall': recall}
    return metrics

def main(train_path=None, 
         test_path=None,
         task=None,
         output_path=None, 
         batch_size=8, 
         num_train_epochs=5):
    print(f'Task: {task}')

    # id2label label2id
    task_id2label = dict()
    task_label2id = dict()

    for idx, loc in enumerate(dict_labels[task]):
        task_id2label[idx] = loc
        task_label2id[loc] = idx

    biobert_model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                            num_labels=len(dict_labels[task]),
                                                            id2label=task_id2label,
                                                            label2id=task_label2id).to(device)
    args = TrainingArguments(
        output_dir = os.path.join(output_path, f"training_{task}"),
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        num_train_epochs= num_train_epochs,
        metric_for_best_model=metric_name,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        report_to="none",
    )

    train_encoded_dataset, test_encoded_dataset = dataset(train_path, test_path, task)

    trainer = Trainer(
        biobert_model,
        args,
        train_dataset= train_encoded_dataset,
        eval_dataset= test_encoded_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    # trainer.save_model(output_path)

    # Get the current date and time
    current_datetime = datetime.now()

    # Format date and time together as a string
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')  # YYYY-MM-DD HH:MM:SS format

    # print("Formatted Date and Time:", formatted_datetime)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    torch.save(biobert_model, os.path.join(output_path, f'{task}.pth'))

if __name__ == "__main__":

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default='./data/large_context_corpus_train.xlsx', required=False, type=str,
                    help="file path for train set")
    ap.add_argument("--test", default='./data/large_context_corpus_test.xlsx', required=False, type=str,
                    help="file path for test set")
    ap.add_argument("--task", default = 'location', required=False, type=str, help="Task name")
    ap.add_argument("--output", default = './saved_models/base', required=False, type=str, help="output path")
    ap.add_argument("--batch", default = 2, required=False, type=int, help="batch size")
    ap.add_argument("--epoch", default = 1, required=False, type=int, help="number of train epochs")

    args = ap.parse_args()
    
    #input_path = args.input
    train_path = args.train
    test_path = args.test
    task = args.task
    output_path = args.output
    batch_size = args.batch
    num_train_epochs = args.epoch

    main(train_path, test_path, task, output_path, batch_size, num_train_epochs)

