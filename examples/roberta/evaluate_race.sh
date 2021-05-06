DATA_DIR=./processed_race       # data directory used during training
MODEL_PATH=./checkpoints/checkpoint_best.pt  # path to the finetuned model checkpoint
PREDS_OUT=preds.tsv                     # output file path to save prediction
TEST_SPLIT=test                         # can be test (Middle) or test1 (High)
fairseq-validate \
    $DATA_DIR \
    --valid-subset $TEST_SPLIT \
    --path $MODEL_PATH \
    --batch-size 1 \
    --task sentence_ranking \
    --criterion sentence_ranking \
    --save-predictions $PREDS_OUT
