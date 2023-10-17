#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

MODE=$1
EXPNAME=$2

split='training'
scenes='cables_1 camera_shake_3 einstein_global_light_changes_1 kidnap_1 motion_1'

#split='test'
#scenes='table_scene reflective_2 planar_1 buddha motion_2 lamp drone kidnap_2 motion_3 cables_4 desk_changing_2 table_2 desk_2 table_global_light_changes large_loop_3 table_5 table_local_light_changes desk_global_light_changes desk_1 cables_5 large_loop_2 helmet scale_change dino trashbin table_1 motion_4 table_6 boxes foreground_occlusion large_non_loop desk_ir_light'

OUT_DIR=/home/yzhang/exps/goslam/eth3d
DATAROOT=/home/yzhang/data/ETH3D/${split}


echo "Start evaluating on ETH3D dataset..."

for sc in ${scenes};
do
  echo Running on $sc ...
  python run.py configs/ETH3D/eth3d.yaml \
                --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME \
                --only_tracking \
                --calibration_txt ${DATAROOT}/${sc}/calibration.txt \
                --input_folder ${DATAROOT}/${sc}
  echo $sc done!
done

echo Results for all scenes are:


if [[ $split == "test" ]]
then
  for sc in ${scenes}
  do
    echo
    mkdir -p ${OUT_DIR}/slam/
    echo copy ${sc} ...
    cp ${OUT_DIR}/${sc}/${EXPNAME}/submission.txt ${OUT_DIR}/slam/${sc}.txt
    touch ${OUT_DIR}/slam/${sc}_runtime.txt
    echo 0.8 >> ${OUT_DIR}/slam/${sc}_runtime.txt
    echo ... OK!
  done
else
  for sc in ${scenes}
  do
    echo
    echo For ${sc}:
    cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
    echo
    cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt
  done
fi

echo All Done!
