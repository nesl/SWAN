#!/bin/bash
trap "kill 0" SIGINT
CUDA_VISIBLE_DEVICES=0 bash jason_scripts/ECCV_test_corruptions.sh 4 6 &
CUDA_VISIBLE_DEVICES=1 bash jason_scripts/ECCV_test_corruptions.sh 8 10 &
CUDA_VISIBLE_DEVICES=2 bash jason_scripts/ECCV_test_corruptions.sh 12 14 &
CUDA_VISIBLE_DEVICES=3 bash jason_scripts/ECCV_test_corruptions.sh 16

wait
echo "Done"