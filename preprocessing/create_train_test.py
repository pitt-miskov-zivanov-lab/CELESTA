import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
    
from labels_loader import *


def create_subset_dataset(large_context_corpus_df, labels = ['location',
       'cell_line', 'cell_type', 'organ', 'disease', 'species'], cols_values_lst = cols_values_lst):

    tmp_lst = set()

    # for each task, keep only the values in the list
    for idx, i in enumerate(labels):
        print(i)

        final_index =  large_context_corpus_df[large_context_corpus_df[i].isin(cols_values_lst[idx])].index.to_list()
        tmp_lst.update(final_index)

    # return row ids
    return tmp_lst 
    
def train_test_from_subset(large_context_corpus_df, output_path, labels=context_types):


    tmp_lst = create_subset_dataset(large_context_corpus_df, labels = labels)
    stacked_df = large_context_corpus_df.iloc[list(tmp_lst)]
    stacked_df.reset_index(names = 'original_index', inplace = True)

    # assign values not in the values list as nan
    before_stacked_df = stacked_df.copy()
    for idx, i in enumerate(labels):
        print(i)
        stacked_df.loc[stacked_df[~stacked_df[i].isin(cols_values_lst[idx])].index.to_list(), i] = np.nan


    for idx, i in enumerate(labels):
        # print(i)
        print(before_stacked_df[i].value_counts())
        print('___ after _____')
        print(stacked_df[i].value_counts())
        print()

    # separate into 6 groups by number of non nan values per row
    six_rows =[]
    five_rows =[]
    four_rows = []
    three_rows = []
    two_rows = []
    one_rows = []
    zero_rows = []

    for i,data in stacked_df[labels].iterrows():
        # print(data.isna())
    
        non_na_sum = (len(labels) - data.isna().sum())
        # print(non_na_sum )
        if  non_na_sum == 6:
            six_rows.append(i)
        elif non_na_sum == 5:
            five_rows.append(i)
        elif non_na_sum == 4:
            four_rows.append(i)
        elif non_na_sum == 3:
            three_rows.append(i)
        elif non_na_sum == 2:
            two_rows.append(i)
        elif non_na_sum == 1:
            one_rows.append(i)
        elif non_na_sum == 0:
            zero_rows.append(i)


    # sample 30% for test set
    test_set = []
    for i in [six_rows, five_rows, four_rows, three_rows, two_rows, one_rows]:
        test_set.extend(pd.Series(i).sample(frac = 0.3, random_state = 0))

    # test set
    large_context_corpus_test_df =  stacked_df.iloc[test_set]
    print('test set shape:', large_context_corpus_test_df.shape )
    large_context_corpus_test_df.to_excel(os.path.join(output_path,'large_context_corpus_test_may4.xlsx'), index=False)

    # train set
    large_context_corpus_train_df = stacked_df[~stacked_df.index.isin(large_context_corpus_test_df.index)]
    print('train set shape:', large_context_corpus_train_df.shape )
    large_context_corpus_train_df.to_excel(os.path.join(output_path,'large_context_corpus_train_may4.xlsx'), index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default = '/ihome/nmiskov-zivanov/tht42/Biomedical-Context-Classification-main/data/large_context_corpus.xlsx', required=False, type=str, help="input file path for data")
    ap.add_argument("--output", default = '', required=False, type=str, help="output path")
    args = ap.parse_args()

    input_path = args.input
    output_path = args.output

    large_context_corpus_df = pd.read_excel(input_path)
    train_test_from_subset(large_context_corpus_df, output_path=output_path)
