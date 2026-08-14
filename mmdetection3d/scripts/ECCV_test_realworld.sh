
corruption='none'
BUDGETS=(16 8 6 4)
for budget in "${BUDGETS[@]}"; do
    python3 tools/test.py /workspace/mmdetection3d/projects/CMT/configs/ECCV_Configs_Rain_Night/FT_Hard_Pruning_EE_Universal_Controller.py \
            /workspace/mmdetection3d/work_dirs/Hard_Pruning_EE_Universal_Controller/epoch_16.pth \
            --cfg-options model.controller.layer_budgets="[$budget]" \
            test_dataloader.dataset.corruptions="[$corruption]" \
            --work-dir ./work_dirs/FT_EE_${corruption}/FT_EE_Controller_$budget \
            > ./work_dirs/Pruner_Rain_${budget}_${corruption}.txt
done