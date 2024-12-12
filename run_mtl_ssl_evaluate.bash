#!/bin/bash
echo -e "\n\nEvaluating mtl\n"

INPUTPATH="./saved_models/mtl"
OUTPUTPATH="./evaluated_results/mtl/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="mtl_fullysupervised" --batch=8 --eval_mtl=true
done

echo -e "\n\nEvaluating mtl+vat\n"

INPUTPATH="./saved_models/mtl+vat"
OUTPUTPATH="./evaluated_results/mtl+vat/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="mtl_vat" --batch=8 --eval_mtl=true
done

echo -e "\n\nEvaluating mtl+fixmatch\n"

INPUTPATH="./saved_models/mtl+fixmatch"
OUTPUTPATH="./evaluated_results/mtl+fixmatch/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="mtl_fixmatch" --batch=8 --eval_mtl=true
done

echo -e "\n\nEvaluating mtl+msp+vat\n"

INPUTPATH="./saved_models/mtl+msp+vat"
OUTPUTPATH="./evaluated_results/mtl+msp+vat/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="mtl_vat" --batch=8 --eval_mtl=true
done

echo -e "\n\nEvaluating mtl+msp+fixmatch\n"

INPUTPATH="./saved_models/mtl+msp+fixmatch"
OUTPUTPATH="./evaluated_results/mtl+msp+fixmatch/"
TASK=("location" "cell_line" "cell_type" "organ" "disease")

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python semi_evaluation.py --input="${INPUTPATH}" --output="${OUTPUTPATH}" --task="$task" --alg="mtl_fixmatch" --batch=8 --eval_mtl=true
done

