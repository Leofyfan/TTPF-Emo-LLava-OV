#!/bin/bash

# projector model path
declare -a MODEL_PATHS=(
    "/root/autodl-tmp/model/mnt/llava-ov-checkpoints/llavaov_finetune_meld_data_20250415_001835"
)

# outputfile path
declare -a OUTPUT_FILES=(
    "/root/project/llava/TTPF-Emo-LLava-OV/eval/results/bigate_new_ttpf_mlp_nocross_flat_16frames_meld_test_results_balanced_15_20.json"
)

# script running
for ((i=0; i<${#MODEL_PATHS[@]}; i++)); do
    echo "evaluating model: ${MODEL_PATHS[$i]}"
    
    # p
    python /root/project/llava/TTPF-Emo-LLava-OV/llava/train/meld_benchmark.py \
        --base_model_path "${MODEL_PATHS[$i]}" \
        --output_file "${OUTPUT_FILES[$i]}"
    
    echo "result saved: ${OUTPUT_FILES[$i]}"
    echo "----------------------------------------"
done

echo "Evaluate Finished!"