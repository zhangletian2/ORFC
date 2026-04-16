for QP in 12 17 22 25 27 30 32
do
    echo "QP=${QP}"
    python tools/dinov2_seg_pipeline.py replay \
        --model vitg14 \
        --feature_root features/voc2012_100/dinov2_vitg14/decoded/vtm/${QP} \
            --layer blk09 \
            --image_list utils/voc2012_val_100.txt
done
for QP in 12 15 17 20 22 32
do
    echo "QP=${QP}"
    python tools/dinov2_seg_pipeline.py replay \
        --model vitg14 \
        --feature_root features/voc2012_100/dinov2_vitg14/decoded/vtm/${QP} \
            --layer blk19 \
            --image_list utils/voc2012_val_100.txt
done
for QP in 0 2 5 7 10 12 22 32
do
    echo "QP=${QP}"
    python tools/dinov2_seg_pipeline.py replay \
        --model vitg14 \
        --feature_root features/voc2012_100/dinov2_vitg14/decoded/vtm/${QP} \
            --layer blk29 \
            --image_list utils/voc2012_val_100.txt
done