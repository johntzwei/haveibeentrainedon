WORKING_DIR=/home/johnny/gpt-neox/haveibeentrainedon/wikitext/pilot/

python tools/preprocess_data.py \
            --input $WORKING_DIR/data/wikitext_perturbed_seed:10_n:100.jsonl \
            --output-prefix ./data/wikitext_perturbed \
            --vocab ./data/gpt2-vocab.json \
            --merge-file ./data/gpt2-merges.txt \
            --dataset-impl mmap \
            --tokenizer-type GPT2BPETokenizer \
            --append-eod \
            --workers 8
