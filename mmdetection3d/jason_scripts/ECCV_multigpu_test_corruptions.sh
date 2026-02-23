#!/bin/bash
trap "kill 0" SIGINT
CUDA_VISIBLE_DEVICES=1 bash jason_scripts/ECCV_test_corruptions.sh 4 6 8  &
CUDA_VISIBLE_DEVICES=2 bash jason_scripts/ECCV_test_corruptions.sh 10 12 &
CUDA_VISIBLE_DEVICES=3 bash jason_scripts/ECCV_test_corruptions.sh 14 16

wait
echo "Done"
