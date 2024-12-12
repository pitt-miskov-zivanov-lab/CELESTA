import os
import pandas as pd
import re
import copy
def convert_df_EDA_txt(input_path, output_path):
    import pandas as pd
    train_path = os.path.join(input_path, 'large_context_corpus_train.xlsx')
    train = pd.read_excel(train_path)

    output_f = os.path.join(output_path, 'train_UMLS_EDA.txt')
    train[['original_index', 'original_index','text', ]].to_csv(output_f, index=False, sep='|', header=False)

def convert_EDA_txt_df(input_path, output_path):
    train_path = os.path.join(input_path, 'large_context_corpus_train.xlsx')
    train_df = pd.read_excel(train_path)
    train_eda_df = copy.deepcopy(train_df)
    #df = pd.read_csv(input_f, sep = '|', header = None)

    input_f = os.path.join(input_path, 'eda_train_UMLS_EDA.txt')
    lines = open(input_f, 'r').readlines()

    original_index = train_eda_df['original_index'].tolist()
    aug_cols = ['umls', 'sr', 'ri', 'rs', 'rd', 'original']

    # TODO: better parsing
    aug_dict = {str(idx): [] for idx in original_index}
    for i, line in enumerate(lines):
        if re.search("^##|^\s+$",line):
            raise ValueError('augmented data was not found')

        parts = line.rstrip().split('|')
        o_idx = parts[0]
        label = parts[1]
        sentence = parts[2]

        aug_dict[o_idx].append(sentence)

    for i in range(len(train_eda_df)):
        o_idx = str(original_index[i])

        for col in aug_cols:
            idx = aug_cols.index(col)
            train_eda_df.loc[i, col] = aug_dict[o_idx][idx]

    # save
    train_eda_df.to_excel(output_path + 'large_context_corpus_train_eda.xlsx', index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = '../data/', required=False, type=str, help="input file path for data")
    ap.add_argument("--output", default = '../data/', required=False, type=str, help="output path")
    ap.add_argument("--txt_to_df", required=False, default = False, help="which function to run, True or False")
    args = ap.parse_args()

    input_path = args.input
    output_path = args.output
    txt_to_df = args.txt_to_df
    
    if txt_to_df:
        convert_EDA_txt_df(input_path, output_path)
    else:
         convert_df_EDA_txt(input_path, output_path)
