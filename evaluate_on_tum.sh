#!/bin/bash

MODE=$1
EXPNAME=$2

OUT_DIR=/home/yzhang/exps/goslam/tum

scenes='360 desk desk2 floor plant room rpy teddy xyz'

echo "Start evaluating on TUM dataset..."

for sc in ${scenes};
do
  echo Running on $sc ...
  python run.py configs/TUM_RGBD/freiburg1_${sc}.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME --only_tracking
  # if [[ $MODE == "mono" ]]
  # then
  #   python run.py configs/TUM_RGBD/scene${sc}_mono.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME --only_tracking
  # else
  #   python run.py configs/TUM_RGBD/scene${sc}.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME --only_tracking
  # fi
  echo $sc done!
done

echo Results for all scenes are:

for sc in ${scenes}
do
  echo
  echo For ${sc}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
  echo
  # cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt
done

echo All Done!
