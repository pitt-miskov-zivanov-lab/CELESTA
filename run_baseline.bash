#!/bin/bash
TRAINPATH="./data/large_context_corpus_train.xlsx"
TESTPATH="./data/large_context_corpus_test.xlsx"
OUTPUTPATH="./saved_models/base/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nTesting bio-context classification\n"

for task in "${TASK[@]}"; do
    echo -e "running task: $task"
    python train_baseline.py --train="${TRAINPATH}" \
    --test="${TESTPATH}"\
    --task="$task" \
    --output="${OUTPUTPATH}" \
    --batch=8 \
    --epoch=20 
done

echo -e "\n\nEvaluating\n"

OUTPUTPATH2="./evaluated_results/base"

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python evaluation.py --input="./saved_models/base/"$task".pth" \
    --task="$task" \
    --output="${OUTPUTPATH2}" \
    --batch=8 \
done