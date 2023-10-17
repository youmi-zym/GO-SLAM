#!/bin/bash

MODE=$1
EXPNAME=$2

OUT_DIR=/home/yzhang/exps/goslam/Replica/

scenes="office0 office1 office2 office3 office4 room0 room1 room2"

echo "Start evaluating on Replica dataset..."

for sc in ${scenes}
do
  echo Running on $sc ...
  if [[ $MODE == "mono" ]]
  then
    python run.py configs/Replica/${sc}_mono.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME
  else
    python run.py configs/Replica/${sc}.yaml --device cuda:0 --mode $MODE --output ${OUT_DIR}/${sc}/$EXPNAME
  fi
  echo $sc done!
done

echo Results for all scenes are:

SUMMARY=${OUT_DIR}/${sc}/${EXPNAME}/summary.txt

for sc in ${scenes}
do
  echo
  echo For ${sc}:
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt
  echo
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt

  echo >> $SUMMARY
  echo For ${sc}: >> $SUMMARY
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_traj.txt >> $SUMMARY
  echo >> SUMMARY
  cat ${OUT_DIR}/${sc}/${EXPNAME}/metrics_mesh.txt >> $SUMMARY
done

echo All Done!

echo All Done! >> $SUMMARY
