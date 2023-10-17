#!/bin/bash

MODE=$1
EXPNAME=$2

OUT_DIR=/home/yzhang/exps/goslam/scannet

# scenes='0000 0059 0106 0169 0181 0054 0233 0465'
sc=0054
fragments=300
dataset_len=6629

echo "Start evaluating on ScanNet dataset..."

for ((i=${fragments}; i<=${dataset_len}+${fragments}-1; i=i+${fragments}));
do
  echo Running on scene ${sc} with fragment ${i} ...
  if [[ $MODE == "mono" ]]
  then
    python run.py configs/ScanNet/scene${sc}_mono.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME/${i} --max_frames ${i}
  else
    python run.py configs/ScanNet/scene${sc}.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME/${i} --max_frames ${i}
  fi
  echo Fragment ${i} done!
done

for ((i=${fragments}; i<=${dataset_len}+${fragments}-1; i=i+${fragments}));
do
  echo
  echo For ${sc} with ${i}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/${i}/metrics_traj.txt
  echo
done

echo All Done!
