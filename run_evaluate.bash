#!/bin/bash
TASK=("location" "cell_line" "cell_type" "organ" "disease")

echo -e "\n\nEvaluating\n"

INPUTPATH='./saved_models/base/'
OUTPUTPATH="./evaluated_results/base/"

for task in "${TASK[@]}"; do
    echo -e "evaluating task: $task"
    python evaluation.py --input="${INPUTPATH}"$task".pth" --task="$task" --output="${OUTPUTPATH}" --batch=8 
done