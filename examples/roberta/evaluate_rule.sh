DATA_DIR=proofwriter-bin/d0_100                    # data directory used during training
MODEL_PATH=checkpoints/d0_100/checkpoint_best.pt  # path to the finetuned model checkpoint
#DATA_DIR=ruletaker-bin                     # data directory used during training
#MODEL_PATH=checkpoints/checkpoint_best.pt  # path to the finetuned model checkpoint
PREDS_OUT=preds_d5.tsv                     # output file path to save prediction
TEST_SPLIT=test                            # can be test (Middle) or test1 (High)
fairseq-validate \
    $DATA_DIR \
    --valid-subset $TEST_SPLIT \
    --path $MODEL_PATH \
    --batch-size 4 \
    --task sentence_prediction \
    --criterion sentence_prediction \
    #--save-predictions $PREDS_OUT


