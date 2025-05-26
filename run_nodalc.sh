#!/bin/bash

SOURCE_YEAR=$1
DATA_DIR="/home/wang/Data/android"
LOG_DIR="/home/wang/Data/dalc_log"

for TARGET_YEAR in {2012..2022}
do
    if [ "$TARGET_YEAR" -eq "$SOURCE_YEAR" ]; then
        continue
    fi

    MODEL_PATH="$DATA_DIR/${SOURCE_YEAR}-${TARGET_YEAR}-nodalc_new.bin"
    SOURCE_FILE="$DATA_DIR/${SOURCE_YEAR}.txt"
    TARGET_FILE="$DATA_DIR/${SOURCE_YEAR}-${TARGET_YEAR}.txt"
    PRED_FILE="$DATA_DIR/${SOURCE_YEAR}-${TARGET_YEAR}-nodalc_new.pred"

    nohup python dalc_learn.py \
        --model="$MODEL_PATH" \
        --nodalc \
        "$SOURCE_FILE" "$TARGET_FILE" \
        > "$LOG_DIR/nodalc_learn_${SOURCE_YEAR}_${TARGET_YEAR}.log" 2>&1 &


    nohup bash -c "
        echo 'Waiting for model $MODEL_PATH...';
        while [ ! -f '$MODEL_PATH' ]; do
            sleep 60
        done
        echo 'Model $MODEL_PATH found. Starting classification.';
        python dalc_classify.py \
            --model=$MODEL_PATH \
            --pred=$PRED_FILE \
            --nodalc \
            $SOURCE_FILE \
            $TARGET_FILE \
            > $LOG_DIR/nodalc_classify_${SOURCE_YEAR}_${TARGET_YEAR}.log 2>&1
    " &
done
