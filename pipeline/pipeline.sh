WORKING_DIR=/home/johnny/gpt-neox
DATASET=$WORKING_DIR/haveibeentrainedon/contexts/17e7_tokens_perturbed.jsonl
# idx, text, sub_idx, original, synonym
PROP_INPUTS=$WORKING_DIR/haveibeentrainedon/contexts/propagation_inputs.csv
CUDA_VISIBLE_DEVICES=2,3,4,5
EXP_DIR=simple

rm -Ir $WORKING_DIR/data/pipeline/*
rm -Ir $WORKING_DIR/runs/70M_cp/*

python $WORKING_DIR/tools/preprocess_data.py \
            --input $DATASET \
            --output-prefix $WORKING_DIR/data/pipeline/pipeline \
            --vocab $WORKING_DIR/data/gpt2-vocab.json \
            --merge-file $WORKING_DIR/data/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 8

python $WORKING_DIR/deepy.py $WORKING_DIR/train.py \
      -d $WORKING_DIR/haveibeentrainedon/pipeline 70M.yml local_setup.yml

python $WORKING_DIR/tools/convert_module_to_hf.py \
    --input_dir $WORKING_DIR/runs/70M_cp/global_step839/ \
    --config_file $WORKING_DIR/haveibeentrainedon/pipeline/70M.yml \
    --output_dir $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR

# score
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES papermill $WORKING_DIR/haveibeentrainedon/pipeline/score_gptneox.ipynb $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR/scoring_output.ipynb \
    -p model_name $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR \
    -p model_precision float32 \
    -p input_fn $PROP_INPUTS \
    -p output_fn $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR/scores.csv 

# calculate propagation rate
papermill $WORKING_DIR/haveibeentrainedon/pipeline/calculate_propagation_rates.ipynb $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR/propagation_output.ipynb \
    -p input_fn $PROP_INPUTS \
    -p scores_fn $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR/scores.csv 
