TOTAL_NUM_UPDATES=7812
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
HEAD_NAME=d0_100     # Custom name for the classification head.
NUM_CLASSES=3           # Number of classes for the classification task.
MAX_SENTENCES=4         # Batch size.
ROBERTA_PATH=./checkpoints/proofonlyd3/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=3,4 fairseq-train proofwriter-bin/d0_100 \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 4 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 4 \
    --save-dir ./checkpoints/d0_100
