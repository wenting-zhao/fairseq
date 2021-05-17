MAX_EPOCH=5           # Number of training epochs.
LR=1e-05              # Peak LR for fixed LR scheduler.
NUM_CLASSES=4
MAX_SENTENCES=1       # Batch size per GPU.
UPDATE_FREQ=8         # Accumulate gradients to simulate training on 8 GPUs.
DATA_DIR=./processed_race
ROBERTA_PATH=./roberta.large/model.pt

CUDA_VISIBLE_DEVICES=2,3 fairseq-train $DATA_DIR --ddp-backend=legacy_ddp \
    --restore-file $ROBERTA_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --task sentence_ranking \
    --num-classes $NUM_CLASSES \
    --init-token 0 --separator-token 2 \
    --max-option-length 128 \
    --max-positions 512 \
    --shorten-method "truncate" \
    --arch roberta_large \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --criterion sentence_ranking \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler fixed --lr $LR \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --batch-size $MAX_SENTENCES \
    --required-batch-size-multiple 1 \
    --update-freq $UPDATE_FREQ \
    --max-epoch $MAX_EPOCH
