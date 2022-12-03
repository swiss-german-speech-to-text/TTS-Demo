#!/bin/bash
cd /scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
ivector-mean ark:dump/22k/mfcc/test/spk2utt scp:dump/22k/xvector/test/xvector.scp ark,scp:dump/22k/xvector/test/spk_xvector.ark,dump/22k/xvector/test/spk_xvector.scp ark,t:dump/22k/xvector/test/num_utts.ark 
EOF
) >dump/22k/xvector/test/log/speaker_mean.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>dump/22k/xvector/test/log/speaker_mean.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( ivector-mean ark:dump/22k/mfcc/test/spk2utt scp:dump/22k/xvector/test/xvector.scp ark,scp:dump/22k/xvector/test/spk_xvector.ark,dump/22k/xvector/test/spk_xvector.scp ark,t:dump/22k/xvector/test/num_utts.ark  ) &>>dump/22k/xvector/test/log/speaker_mean.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>dump/22k/xvector/test/log/speaker_mean.log
echo '#' Accounting: end_time=$time2 >>dump/22k/xvector/test/log/speaker_mean.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>dump/22k/xvector/test/log/speaker_mean.log
echo '#' Finished at `date` with status $ret >>dump/22k/xvector/test/log/speaker_mean.log
[ $ret -eq 137 ] && exit 100;
touch dump/22k/xvector/test/q/done.129530
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -p gpu --gres=gpu:4 -c 4 --cpus-per-task 8 --time 24:00:00 --qos 1day --mem-per-cpu 32G --partition a100  --open-mode=append -e dump/22k/xvector/test/q/speaker_mean.log -o dump/22k/xvector/test/q/speaker_mean.log  /scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/dump/22k/xvector/test/q/speaker_mean.sh >>dump/22k/xvector/test/q/speaker_mean.log 2>&1
