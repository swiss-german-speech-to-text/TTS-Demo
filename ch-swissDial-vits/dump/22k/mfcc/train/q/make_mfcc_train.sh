#!/bin/bash
cd /scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
compute-mfcc-feats --write-utt2dur=ark,t:dump/22k/mfcc/train/log/utt2dur.${SLURM_ARRAY_TASK_ID} --verbose=2 --config=conf/mfcc.conf scp,p:dump/22k/mfcc/train/log/wav_train.${SLURM_ARRAY_TASK_ID}.scp ark:- | copy-feats --write-num-frames=ark,t:dump/22k/mfcc/train/log/utt2num_frames.${SLURM_ARRAY_TASK_ID} --compress=true ark:- ark,scp:/scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/dump/22k/mfcc/train/data/raw_mfcc_train.${SLURM_ARRAY_TASK_ID}.ark,/scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/dump/22k/mfcc/train/data/raw_mfcc_train.${SLURM_ARRAY_TASK_ID}.scp 
EOF
) >dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( compute-mfcc-feats --write-utt2dur=ark,t:dump/22k/mfcc/train/log/utt2dur.${SLURM_ARRAY_TASK_ID} --verbose=2 --config=conf/mfcc.conf scp,p:dump/22k/mfcc/train/log/wav_train.${SLURM_ARRAY_TASK_ID}.scp ark:- | copy-feats --write-num-frames=ark,t:dump/22k/mfcc/train/log/utt2num_frames.${SLURM_ARRAY_TASK_ID} --compress=true ark:- ark,scp:/scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/dump/22k/mfcc/train/data/raw_mfcc_train.${SLURM_ARRAY_TASK_ID}.ark,/scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/dump/22k/mfcc/train/data/raw_mfcc_train.${SLURM_ARRAY_TASK_ID}.scp  ) &>>dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>dump/22k/mfcc/train/log/make_mfcc_train.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch dump/22k/mfcc/train/q/done.70341.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -p gpu --gres=gpu:4 -c 4 --cpus-per-task 8 --time 24:00:00 --qos 1day --mem-per-cpu 32G --partition a100  --open-mode=append -e dump/22k/mfcc/train/q/make_mfcc_train.log -o dump/22k/mfcc/train/q/make_mfcc_train.log --array 1-32 /scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/dump/22k/mfcc/train/q/make_mfcc_train.sh >>dump/22k/mfcc/train/q/make_mfcc_train.log 2>&1
