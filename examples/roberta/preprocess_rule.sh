fairseq-preprocess \
    --only-source \
    --trainpref "proofwriter/d0_1000/train.input0.bpe" \
    --validpref "proofwriter/d0_1000/dev.input0.bpe" \
    --testpref "proofwriter/d0_1000/test.input0.bpe" \
    --destdir "proofwriter-bin/d0_1000/input0" \
    --workers 60 \
    --srcdict dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "rawproof/d0_1000/train.label" \
    --validpref "rawproof/d0_1000/dev.label" \
    --testpref "rawproof/d0_1000/test.label" \
    --destdir "proofwriter-bin/d0_1000/label" \
    --workers 60
