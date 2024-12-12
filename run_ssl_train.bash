#!/bin/bash
#vat
TRAINPATH="./data/large_context_corpus_train.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH="./saved_models/vat"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting semi-supervised learning bio-context classification\n"
echo -e "Algorithm: vat\n"

for task in "${TASK[@]}"; do
    echo -e "running task: $task"
    python semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="vat" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done

#ood_vat
TRAINPATH="./data/large_context_corpus_train_ood.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH="./saved_models/vat_ood"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting open-set semi-supervised learning bio-context classification\n"
echo -e "Algorithm: vat\n"

for task in "${TASK[@]}"; do
    echo -e "running task: $task"
    python semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="vat" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done

#######################################################
#fixmatch
TRAINPATH="./data/large_context_corpus_train_eda.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH="./saved_models/fixmatch"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting semi-supervised learning bio-context classification\n"
echo -e "Algorithm: fixmatch\n"

for task in "${TASK[@]}"; do
    echo -e "running task: $task"
    python semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="fixmatch" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done

#ood_fixmatch
TRAINPATH="./data/large_context_corpus_train_eda_ood.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH="./saved_models/fixmatch_ood"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting open-set semi-supervised learning bio-context classification\n"
echo -e "Algorithm: fixmatch\n"

for task in "${TASK[@]}"; do
    echo -e "running task: $task"
    python semi_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="fixmatch" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done