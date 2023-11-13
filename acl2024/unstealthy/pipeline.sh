WORKING_DIR=/home/johnny/gpt-neox
DATASET=$WORKING_DIR/haveibeentrainedon/unstealthy/wikitext_perturbed_substitutions.jsonl
# PROP_INPUTS=$WORKING_DIR/haveibeentrainedon/contexts/propagation_inputs.csv
EXP_DIR=wikitext_substitutions
MODEL_YML=70M.yml
CUDA_HOME=$CONDA_PREFIX/

read -p 'Preprocess [y|n]: ' do_preprocess
read -p 'Train [y|n]: ' do_train
read -p 'Score [y|n]: ' do_score

if [ $do_preprocess == "y" ]; then
    rm -r $WORKING_DIR/data/pipeline/*
    python $WORKING_DIR/tools/preprocess_data.py \
                --input $DATASET \
                --output-prefix $WORKING_DIR/data/pipeline/pipeline \
                --vocab $WORKING_DIR/data/gpt2-vocab.json \
                --merge-file $WORKING_DIR/data/gpt2-merges.txt \
                --dataset-impl mmap \
                --tokenizer-type GPT2BPETokenizer \
                --append-eod \
                --workers 8
fi

if [ $do_train == "y" ]; then
    rm -r $WORKING_DIR/runs/70M_cp/*

    python $WORKING_DIR/deepy.py $WORKING_DIR/train.py \
          -d $WORKING_DIR/haveibeentrainedon/pipeline $MODEL_YML local_setup.yml

    python $WORKING_DIR/tools/convert_module_to_hf.py \
        --input_dir $WORKING_DIR/runs/70M_cp/$ckpt/ \
        --config_file $WORKING_DIR/haveibeentrainedon/pipeline/$MODEL_YML \
        --output_dir $WORKING_DIR/haveibeentrainedon/pipeline/$EXP_DIR
fi

if [ $do_score == "y" ]; then
    ckpt="$(< ${WORKING_DIR}/runs/70M_cp/latest)"

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
fi
