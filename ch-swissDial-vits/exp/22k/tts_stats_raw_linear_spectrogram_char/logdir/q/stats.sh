#!/bin/bash
cd /scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  set | grep SLURM | while read line; do echo "# $line"; done
  echo -n '# '; cat <<EOF
python3 -m espnet2.bin.gan_tts_train --collect_stats true --write_collected_feats false --use_preprocessor true --token_type char --token_list dump/22k/token_list/char/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --normalize none --pitch_normalize none --energy_normalize none --train_data_path_and_name_and_type dump/22k/raw/train/text,text,text --train_data_path_and_name_and_type dump/22k/raw/train/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/22k/raw/dev/text,text,text --valid_data_path_and_name_and_type dump/22k/raw/dev/wav.scp,speech,sound --train_shape_file exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/train.${SLURM_ARRAY_TASK_ID}.scp --valid_shape_file exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/valid.${SLURM_ARRAY_TASK_ID}.scp --output_dir exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.${SLURM_ARRAY_TASK_ID} --config ./conf/tuning/train+xvector_vits.yaml --feats_extract linear_spectrogram --feats_extract_conf n_fft=1024 --feats_extract_conf hop_length=256 --feats_extract_conf win_length=null --pitch_extract_conf fs=22050 --pitch_extract_conf n_fft=1024 --pitch_extract_conf hop_length=256 --pitch_extract_conf f0max=400 --pitch_extract_conf f0min=80 --energy_extract_conf fs=22050 --energy_extract_conf n_fft=1024 --energy_extract_conf hop_length=256 --energy_extract_conf win_length=null --train_data_path_and_name_and_type dump/22k/xvector/train/xvector.scp,spembs,kaldi_ark --valid_data_path_and_name_and_type dump/22k/xvector/dev/xvector.scp,spembs,kaldi_ark --use_wandb true --resume true --wandb_id vits-ch-ch-char-swissDial 
EOF
) >exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
if [ "$CUDA_VISIBLE_DEVICES" == "NoDevFiles" ]; then
  ( echo CUDA_VISIBLE_DEVICES set to NoDevFiles, unsetting it... 
  )>>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
  unset CUDA_VISIBLE_DEVICES
fi
time1=`date +"%s"`
 ( python3 -m espnet2.bin.gan_tts_train --collect_stats true --write_collected_feats false --use_preprocessor true --token_type char --token_list dump/22k/token_list/char/tokens.txt --non_linguistic_symbols none --cleaner none --g2p none --normalize none --pitch_normalize none --energy_normalize none --train_data_path_and_name_and_type dump/22k/raw/train/text,text,text --train_data_path_and_name_and_type dump/22k/raw/train/wav.scp,speech,sound --valid_data_path_and_name_and_type dump/22k/raw/dev/text,text,text --valid_data_path_and_name_and_type dump/22k/raw/dev/wav.scp,speech,sound --train_shape_file exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/train.${SLURM_ARRAY_TASK_ID}.scp --valid_shape_file exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/valid.${SLURM_ARRAY_TASK_ID}.scp --output_dir exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.${SLURM_ARRAY_TASK_ID} --config ./conf/tuning/train+xvector_vits.yaml --feats_extract linear_spectrogram --feats_extract_conf n_fft=1024 --feats_extract_conf hop_length=256 --feats_extract_conf win_length=null --pitch_extract_conf fs=22050 --pitch_extract_conf n_fft=1024 --pitch_extract_conf hop_length=256 --pitch_extract_conf f0max=400 --pitch_extract_conf f0min=80 --energy_extract_conf fs=22050 --energy_extract_conf n_fft=1024 --energy_extract_conf hop_length=256 --energy_extract_conf win_length=null --train_data_path_and_name_and_type dump/22k/xvector/train/xvector.scp,spembs,kaldi_ark --valid_data_path_and_name_and_type dump/22k/xvector/dev/xvector.scp,spembs,kaldi_ark --use_wandb true --resume true --wandb_id vits-ch-ch-char-swissDial  ) &>>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
ret=$?
sync || true
time2=`date +"%s"`
echo '#' Accounting: begin_time=$time1 >>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: end_time=$time2 >>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
echo '#' Finished at `date` with status $ret >>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/stats.$SLURM_ARRAY_TASK_ID.log
[ $ret -eq 137 ] && exit 100;
touch exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/q/done.1276.$SLURM_ARRAY_TASK_ID
exit $[$ret ? 1 : 0]
## submitted with:
# sbatch --export=PATH  -p gpu --gres=gpu:4 -c 4 --cpus-per-task 8 --time 24:00:00 --qos 1day --mem-per-cpu 32G --partition a100  --open-mode=append -e exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/q/stats.log -o exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/q/stats.log --array 1-32 /scicore/home/graber0001/perity98/IP922/espnet/egs2/ch_swissDial_vits/tts/exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/q/stats.sh >>exp/22k/tts_stats_raw_linear_spectrogram_char/logdir/q/stats.log 2>&1
