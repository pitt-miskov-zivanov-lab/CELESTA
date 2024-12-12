__author__ = 'difei'
__email__ = 'DIT18@pitt.edu'

import os.path
import numpy as np
import pandas as pd
from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semilearn import BasicDataset
import torch
from utils import map_label2ids, split_ssl_list
from preprocessing.labels_loader import *

torch.cuda.empty_cache()


def main(train_path=None,
         test_path=None,
         task=None,
         output_path=None,
         alg=None,
         batch_size=8,
         num_train_epochs=5):
    if task not in context_types:
        raise ValueError(f"{task} is not in pre-defined contexts")

    print(f'Task: {task}')

    train_path = os.path.join(train_path)
    test_path = os.path.join(test_path)

    # read only subset of data
    train_df = pd.read_excel(train_path)[['text', task]]
    test_df = pd.read_excel(test_path)[['text', task]]

    map_label2ids(train_df, task)
    map_label2ids(test_df, task)

    # define configs and create config
    config = {
        'algorithm': alg,
        'net': 'biobert',
        'use_pretrain': False,
        'save_name': f'{task}',
        'save_dir': output_path,
        'pretrain_path': f'{output_path}/{task}/lastest_model.pth',
        'overwrite': True,
        'dataset': None,
        'use_cat': False,
        'use_mtl': False,  # newly added

        # optimization configs
        'epoch': num_train_epochs,
        'num_train_iter': 5000,
        'num_eval_iter': 5000,
        'num_log_iter': 100,
        'optim': 'AdamW',
        'lr': 5e-5,
        'layer_decay': 0.5,
        'batch_size': batch_size,
        'eval_batch_size': batch_size,

        # dataset configs
        'num_classes': num_classes[context_types.index(task)],

        # algorithm specific configs
        'hard_label': True,
        'vat_embed': True,
        'uratio': 1,
        'ulb_loss_ratio': 1.0,

        # device configs
        'gpu': 0,
        'world_size': 1,
        "num_workers": 2,
        'distributed': False,
    }
    config = get_config(config)

    # create model and specify algorithm
    algorithm = get_algorithm(config, get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

    # only keep lb data
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    lb_text_list, lb_label_list, _, _ = split_ssl_list(train_df, task=task, alg=alg)
    test_lb_text_list, test_lb_label_list, _, _ = split_ssl_list(test_df, task=task)

    lb_dataset = BasicDataset(config.algorithm, lb_text_list, lb_label_list, config.num_classes, is_ulb=False)
    eval_dataset = BasicDataset(config.algorithm, test_lb_text_list, test_lb_label_list, config.num_classes,
                                is_ulb=False)

    train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size)
    eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size)

    # training and evaluation
    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, eval_loader=eval_loader)
    trainer.evaluate(eval_loader)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # ap.add_argument("--input", default='./data/', required=False, type=str,
    #                 help="input file path for data")
    ap.add_argument("--train", default='./data/large_context_corpus_train_auto.xlsx', required=False, type=str,
                    help="file path for train set")
    ap.add_argument("--test", default='./data/large_context_corpus_test_auto.xlsx', required=False, type=str,
                    help="file path for test set")
    ap.add_argument("--task", default='disease', required=False, type=str, help="Task name")
    ap.add_argument("--alg", default='fullysupervised', required=False, type=str, help="algorithm name")
    ap.add_argument("--output", default='./saved_models/base', required=False, type=str,
                    help="output path")
    ap.add_argument("--batch", default=8, required=False, type=int, help="batch size")
    ap.add_argument("--epoch", default=1, required=False, type=int, help="number of train epochs")

    args = ap.parse_args()

    # input_path = args.input
    train_path = args.train
    test_path = args.test
    task = args.task
    alg = args.alg
    output_path = args.output
    batch_size = args.batch
    num_train_epochs = args.epoch

    main(train_path, test_path, task, output_path, alg, batch_size, num_train_epochs)
