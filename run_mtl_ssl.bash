#!/bin/bash
#vat 
TRAINPATH="./data/large_context_corpus_train.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH='./saved_models/mtl_vat'
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting multi-tasking semi-supervised learning bio-context classification\n"
echo -e "Algorithm: vat\n"

for task in "${TASK[@]}"; do
    echo -e "running MTL task: $task"
    python mtl_semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="mtl_vat" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done

#ood-vat 
TRAINPATH="./data/large_context_corpus_train_ood.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH='./mtl_vat_ood'
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting open-set multi-tasking semi-supervised learning bio-context classification\n"
echo -e "Algorithm: vat\n"

for task in "${TASK[@]}"; do
    echo -e "running MTL task: $task"
    python mtl_semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="mtl_vat" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done

######################################################
#fixmatch 
TRAINPATH="./data/large_context_corpus_train_eda.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH='./saved_models/mtl_fixmatch'
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting multi-tasking semi-supervised learning bio-context classification\n"
echo -e "Algorithm: fixmatch\n"

for task in "${TASK[@]}"; do
    echo -e "running MTL task: $task"
    python mtl_semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="mtl_fixmatch" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done

#ood-fixmatch
TRAINPATH="./data/large_context_corpus_train_eda_ood.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH='./saved_models/mtl_fixmatch_ood'
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting open-set multi-tasking semi-supervised learning bio-context classification\n"
echo -e "Algorithm: fixmatch\n"

for task in "${TASK[@]}"; do
    echo -e "running MTL task: $task"
    python mtl_semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="mtl_fixmatch" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done