WORKING_DIR=/home/johnny/gpt-neox/

python tools/preprocess_data.py \
            --input $WORKING_DIR/data/pile_17e7/17e7_tokens.jsonl \
            --output-prefix ./data/pile_17e7/17e7_tokens \
            --vocab ./data/gpt2-vocab.json \
            --merge-file ./data/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 20
