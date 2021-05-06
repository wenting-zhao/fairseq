for SPLIT in 'train' 'dev' 'test'; do
    python -m multiprocessing_bpe_encoder \
        --encoder-json encoder.json \
        --vocab-bpe vocab.bpe \
        --inputs "/mnt/beegfs/bulk/mirror/wz346/rule-reasoning-dataset-V2020.2.4/d5/$SPLIT.input0" \
        --outputs "ruletaker/d5/$SPLIT.input0.bpe" \
        --workers 60 \
        --keep-empty
done
