# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/evaluation.py

import os
from .hook import Hook


class EvaluationHook(Hook):
    """
    Evaluation Hook for validation during training
    Add multi-task learning setting
    """

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_eval_iter) or self.is_last_iter(algorithm):
            algorithm.print_fn("validating...")

            eval_dest = 'eval'
            if algorithm.use_mtl:
                eval_dicts = algorithm.evaluate_mtl(eval_dest)

                # FIXME: update average score
                # sum_dict = {
                #     eval_dest + "/loss": 0,
                #     eval_dest + "/top-1-acc": 0,
                #     eval_dest + "/balanced_acc": 0,
                #     eval_dest + "/precision": 0,
                #     eval_dest + "/recall": 0,
                #     eval_dest + "/F1": 0,
                # }

                # for task_name, task_value in eval_dicts.items():
                #     assert eval_dest + "/logits" not in task_value.keys(), "return_logits not implemented in multi-task settings yet"
                #     for key, value in task_value.items():
                #         sum_dict[key] += value
                #
                # eval_dict = {key: value / len(algorithm.num_classes) for key, value in sum_dict.items()}

                # update context score
                tasks = list(eval_dicts.keys())
                task = tasks[0]
                eval_dict = eval_dicts[task]
                algorithm.log_dict.update(eval_dict)
                return eval_dicts
            else:
                eval_dict = algorithm.evaluate(eval_dest)
                algorithm.log_dict.update(eval_dict)
                return eval_dict 

    def after_run(self, algorithm):
        
        if not algorithm.args.multiprocessing_distributed or (algorithm.args.multiprocessing_distributed and algorithm.args.rank % algorithm.ngpus_per_node == 0):
            save_path = os.path.join(algorithm.save_dir, algorithm.save_name)
            algorithm.save_model('latest_model.pth', save_path)

        results_dict = {'eval/best_acc': algorithm.best_eval_acc, 'eval/best_it': algorithm.best_it}
        if 'test' in algorithm.loader_dict:
            # load the best model and evaluate on test dataset
            best_model_path = os.path.join(algorithm.args.save_dir, algorithm.args.save_name, 'model_best.pth')
            algorithm.load_model(best_model_path)
            test_dict = algorithm.evaluate('test')
            results_dict['test/best_acc'] = test_dict['test/top-1-acc']
        algorithm.results_dict = results_dict
        