import torch
import argparse
import os
import copy
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer
import logging

# load utils
import utils
model_name = "dmis-lab/biobert-v1.1"
utils.tokenizer = AutoTokenizer.from_pretrained(model_name)
from utils import get_test_df, get_dataset, get_dataloader, map_ids2label, custom_style

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(model_path, test_path=None, output_path=None, task=None, batch_size=None):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    logging.basicConfig(filename=os.path.join(output_path, f"{task}_log.txt"), filemode="w", format="%(asctime)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    test_df = get_test_df(test_path=test_path, task=task)
    test_dataset = get_dataset(test_df, task=task)
    eval_loader = get_dataloader(test_dataset, batch_size=batch_size)

    # TODO: load_state_dict 
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            batch.to(device)
            outputs = model(**batch)

        logits = outputs.logits
        prediction = logits.argmax(dim=-1)

        y_pred.extend(prediction.tolist())
        y_true.extend(batch["labels"].tolist())

    p, r, f, s = precision_recall_fscore_support(y_pred=y_pred, y_true=y_true, average="micro")
    logger.info("precision: {:.4f}".format(p))
    logger.info("recall: {:.4f}".format(r))
    logger.info("f1: {:.4f}".format(f))

    # save dataframe for error analysis
    pred_output = copy.deepcopy(test_df)
    pred_output[f'{task}_pred'] = y_pred

    map_ids2label(pred_output, task)

    pred_output.style.apply(custom_style, columns=[task, f'{task}_pred'], axis=1).to_excel(os.path.join(output_path, f'{task}_output.xlsx'))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = './saved_models/base/location.pth', required=False, type=str, help="saved model checkpoints")
    ap.add_argument("--test", default='./data/large_context_corpus_test.xlsx', required=False, type=str, help="file path for test set")
    ap.add_argument("--task", default = 'location', required=False, type=str, help="Task name")
    ap.add_argument("--output", default = './evaluated_results/base/', required=False, type=str, help="output path")
    ap.add_argument("--batch", default = 8, required=False, type=int, help="batch size")
    args = ap.parse_args()

    model_path = args.input
    test_path = args.test
    task = args.task
    output_path = args.output
    batch_size = args.batch
    
    main(model_path, test_path, output_path, task=task, batch_size=batch_size)