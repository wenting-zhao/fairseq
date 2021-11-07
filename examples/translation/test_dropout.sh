for f in {1..200}
do
    echo $f
    CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/iwslt14.tokenized.de-en \
        --path checkpoints/checkpoint_best.pt --retain-dropout \
        --batch-size 128 --beam 5 --remove-bpe --seed $f --gen-subset valid\
        --results-path results/best_model_testtime_dropout_valid_run$f
done
