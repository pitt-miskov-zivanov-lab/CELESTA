#!/bin/bash
#vat 
TRAINPATH="./data/large_context_corpus_train.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH='./saved_models/mtl'
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting multi-tasking learning bio-context classification\n"

for task in "${TASK[@]}"; do
    echo -e "running MTL task: $task"
    python mtl_train.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --alg="mtl_fullysupervised" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=50 
done
