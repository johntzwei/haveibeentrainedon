WORKING_DIR=/home/johnny/gpt-neox/haveibeentrainedon/wikitext/original/

python tools/preprocess_data.py \
            --input $WORKING_DIR/data/wikitext.jsonl \
            --output-prefix ./data/wikitext/wikitext \
            --vocab ./data/gpt2-vocab.json \
            --merge-file ./data/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 8
