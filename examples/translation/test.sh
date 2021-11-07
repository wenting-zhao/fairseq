for f in checkpoints/checkpoint[0-9]*.pt
do
    ff=`basename $f`
    echo $f $ff
    echo ${ff:10:-3}
    CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/iwslt14.tokenized.de-en \
        --path $f \
        --batch-size 128 --beam 5 --remove-bpe \
        --results-path results/${ff:10:-3}
done
