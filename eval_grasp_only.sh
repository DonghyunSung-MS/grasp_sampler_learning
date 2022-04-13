#!/bin/bash

N=10000
declare -a object_names=(
004_sugar_box
# 005_tomato_soup_can
# 006_mustard_bottle
# 024_bowl
# 025_mug
)
methods="FLOW_VAE_GAN"

for object_name in ${object_names[@]}
do
    echo "Running GraspOnly ${object_name}"
    declare -a methods=(
    FLOW_scaleexp_N4
    FLOW_scaleexp_N8
    FLOW_scalesigmoidsoftplus_N8
    FLOW_scalesigmoidsoftplus_N4
    FLOW_scalesigmoidsoftplus_N16
    GAN_z2
    GAN_z4
    GAN_z8
    VAE_z2_kl0.01
    VAE_z4_kl0.01
    VAE_z4_kl0.1
    VAE_z4_kl1.0
    VAE_z8_kl0.01
    VAE_z8_kl0.1
    VAE_z8_kl1.0
    VAE_z16_kl1.0
    VAE_z16_kl0.01
    VAE_z16_kl0.1
    )
    for method in ${methods[@]}
    do
        python scripts/eval/grasp_only.py checkpoints/${object_name}/${method} --num_sample=$N --q --offscreen_save &
        # python scripts/eval/grasp_only.py checkpoints/${object_name}/${method} --num_sample=$N --offscreen_save
        # python scripts/eval/grasp_only.py checkpoints/${object_name}/${method} --num_sample=$N --ds
        
    done
done
wait 
echo "All done"
exit 0

for object_name in $object_names
do
    echo "Running GraspOnly ${object_name}"
    python scripts/eval/grasp_only.py checkpoints/${object_name} --save --run_name=${methods}
done