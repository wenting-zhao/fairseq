for SPLIT in 'train' 'dev' 'test'; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "rawproof/d0_1000/$SPLIT.input0" \
        --outputs "proofwriter/d0_1000/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done
