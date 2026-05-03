# for QP in 25 27 30 32 35
# do
#     echo "QP=${QP}"
#     CUDA_VISIBLE_DEVICES=7 python tools/dinov2_seg_pipeline.py replay \
#         --model vitl14 \
#         --feature_root features/voc2012_500/dinov2_vitl14/decoded/vtm/${QP} \
#             --layer blk05 blk10 blk15 \
#             --image_list utils/voc2012_val_500.txt
# done

for QP in 0 2 5 7 10 22
do
    echo "QP=${QP}"
    CUDA_VISIBLE_DEVICES=7 python tools/dinov2_seg_pipeline.py replay \
        --model vitl14 \
        --feature_root features/voc2012_500/dinov2_vitl14/decoded/vtm/${QP} \
            --layer blk20 \
            --image_list utils/voc2012_val_500.txt
done