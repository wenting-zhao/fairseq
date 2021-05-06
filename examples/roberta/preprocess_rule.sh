fairseq-preprocess \
    --only-source \
    --trainpref "ruletaker/d5/train.input0.bpe" \
    --validpref "ruletaker/d5/dev.input0.bpe" \
    --testpref "ruletaker/d5/test.input0.bpe" \
    --destdir "ruletaker-bin/d5/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "/mnt/beegfs/bulk/mirror/wz346/rule-reasoning-dataset-V2020.2.4/d5/train.label" \
    --validpref "/mnt/beegfs/bulk/mirror/wz346/rule-reasoning-dataset-V2020.2.4/d5/dev.label" \
    --testpref "/mnt/beegfs/bulk/mirror/wz346/rule-reasoning-dataset-V2020.2.4/d5/test.label" \
    --destdir "ruletaker-bin/d5/label" \
    --workers 60
