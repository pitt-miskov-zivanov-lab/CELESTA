__author__ = 'difei'
__email__ = 'DIT18@pitt.edu'

import os
import copy
import numpy as np
import pandas as pd
from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, MTLTrainer
from semilearn import BasicDataset
import torch

torch.cuda.empty_cache()

from preprocessing.labels_loader import *
from utils import map_label2ids, split_mtl_ssl_list, split_ssl_list

from semilearn.algorithms.mtl_fullysupervised import MTLFullySupervised

device = torch.device("cuda")

def main(train_path=None,
         test_path=None,
         task=None,
         output_path=None,
         alg=None,
         batch_size=8,
         num_train_epochs=5):
    if task not in context_types:
        raise ValueError(f"{task} is not in pre-defined contexts")

    print(f'MTL Task: {task}')

    train_path = os.path.join(train_path)
    test_path = os.path.join(test_path)

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)[['text', task, 'relation']]

    # define configs and create config
    dir_path = os.path.dirname(output_path)

    tasks_info = [task, 'relation']
    num_classes = [len(task_classes[task]), len(RELATION)]

    config = {
        'algorithm': alg,
        'net': 'biobert',
        'use_pretrain': False,
        'save_name': f'{task}',
        'save_dir': output_path,
        'pretrain_path': f'{output_path}/{task}/lastest_model.pth',
        'overwrite': True,
        'dataset': None,
        'use_mtl': True,

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
        'num_classes': num_classes,
        'tasks_info': tasks_info,

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

    for t in tasks_info:
        map_label2ids(train_df, t)
        map_label2ids(test_df, t)

    train_df.dropna(inplace=True)
    lb_text_list, lb_label_list, ulb_text_list, ulb_label_list = split_mtl_ssl_list(train_df, task=task, alg=alg)

    lb_dataset = BasicDataset(config.algorithm, lb_text_list, lb_label_list, config.num_classes, is_ulb=False)
    train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size)

    evaluate_loaders = []
    for task in tasks_info:
        # avoid SettingWithCopyWarning
        test_df_task = copy.deepcopy(test_df[['text', task]])
        test_df_task.dropna(inplace=True)
        test_lb_text_list, test_lb_label_list, _, _ = split_ssl_list(test_df_task, task=task)
        eval_dataset = BasicDataset(config.algorithm, test_lb_text_list, test_lb_label_list, config.num_classes,
                                    is_ulb=False)
        eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size)
        evaluate_loaders.append(eval_loader)

    # training and evaluation
    mtl_trainer = MTLTrainer(config, algorithm)
    mtl_trainer.fit(train_lb_loader, eval_loaders=evaluate_loaders)
    # mtl_trainer.evaluate(evaluate_loaders)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    # ap.add_argument("--input", default='./data/', required=False, type=str,
    #                 help="input file path for data")
    ap.add_argument("--train", default='./data/large_context_corpus_train.xlsx', required=False, type=str,
                    help="file path for train set")
    ap.add_argument("--test", default='./data/large_context_corpus_test.xlsx', required=False, type=str,
                    help="file path for test set")
    ap.add_argument("--task", default='location', required=False, type=str, help="Task name")
    ap.add_argument("--alg", default='mtl_fullysupervised', required=False, type=str, help="algorithm name")
    ap.add_argument("--output", default='./saved_models/mtl', required=False, type=str,
                    help="output path")
    ap.add_argument("--batch", default=8, required=False, type=int, help="batch size")
    ap.add_argument("--epoch", default=1, required=False, type=int, help="number of train epochs")

    args = ap.parse_args()

    train_path = args.train
    test_path = args.test
    task = args.task
    alg = args.alg
    output_path = args.output
    batch_size = args.batch
    num_train_epochs = args.epoch

    main(train_path, test_path, task, output_path, alg, batch_size, num_train_epochs)
