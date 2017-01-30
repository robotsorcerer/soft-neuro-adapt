#! /bin/bash

TORCH=th
MAIN_PATH=$(rospack find farnn)
MAIN="main.lua"
WITH_PLOT=-plot
WITH_VICON=-vicon
WITH_ROS=-ros

#only run main.lua with default options for now
$TORCH "$MAIN_PATH/$MAIN"
 # $TORCH $MAIN
