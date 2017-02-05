#! /bin/bash

export verbose=true

TORCH=th
FARNN_PATH=$(rospack find farnn)
MODEL="src/network/data_fastlstm-net.t7"
export checkpoint="$FARNN_PATH/$MODEL"
SAMPLE="src/sample.lua"
WITH_VERBOSE=-verbose
WITH_CHECKPOINT=-checkpoint

#only run main.lua with default options for now
$TORCH "$FARNN_PATH/$SAMPLE" "$WITH_VERBOSE" #"$WITH_CHECKPOINT"