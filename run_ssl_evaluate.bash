#!/bin/bash
echo -e "\n\nEvaluating ssl_vat\n"

INPUTPATH="./saved_models/vat"
OUTPUTPATH="./evaluated_results/vat/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="vat" --batch=8
done

echo -e "\n\nEvaluating ssl_fixmatch\n"

INPUTPATH="./saved_models/fixmatch"
OUTPUTPATH="./evaluated_results/fixmatch/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="fixmatch" --batch=8
done

echo -e "\n\nEvaluating ossl_vat\n"

INPUTPATH="./saved_models/msp+vat"
OUTPUTPATH="./evaluated_results/msp+vat/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="vat" --batch=8
done

echo -e "\n\nEvaluating ossl_fixmatch\n"

INPUTPATH="./saved_models/msp+fixmatch"
OUTPUTPATH="./evaluated_results/msp+fixmatch/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="fixmatch" --batch=8
done


