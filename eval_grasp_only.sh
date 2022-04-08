N=10000
object_names="
004_sugar_box
005_tomato_soup_can
006_mustard_bottle
024_bowl
025_mug
"
for object_name in $object_names
do
    echo "Running GraspOnly ${object_name}"
    python scripts/eval/grasp_only.py checkpoints/${object_name}/FLOW_scalesigmoidsoftplus_N8 --num_sample=$N --q
    python scripts/eval/grasp_only.py checkpoints/${object_name}/VAE_z2_kl0.01 --num_sample=$N --q


done
