
melgan_vocoder_file = "../de-ch/exp/train_melgan.v3/checkpoint-640000steps.pkl"


ch_ch_vits_word = {
    "working_dir":  "/ch-ch-vits-word",
    "model_file" : "./exp/22k/tts_train+xvector_vits_raw_word_use_wandbtrue_resumetrue_wandb_idvits-ch-ch-word/993epoch.pth",
    "vocoder_file": ""
}

ch_ch_vits_char = {
    "working_dir":  "/ch-ch-vits",
    "model_file" : "./exp/22k/tts_train+xvector_vits_raw_char_use_wandbtrue_resumetrue_wandb_idvits-ch-ch-char/928epoch.pth",
    "vocoder_file": ""
}

ch_ch_vits_char_reduced = {
    "working_dir":  "/ch-ch-vits",
    "model_file" : "./exp/22k/tts_train+xvector_vits_raw_char_use_wandbtrue_resumetrue_wandb_idvits-ch-ch-char-reduced/721epoch.pth",
    "vocoder_file": ""
}

ch_ch_vits_char_mixed = {
    "working_dir":  "/ch-mixed-vits",
    "model_file" : "./exp/22k/tts_train+xvector_vits_raw_char_use_wandbtrue_resumetrue_wandb_idvits-ch-ch-char-mixed/489epoch.pth",
    "vocoder_file": ""
}

ch_ch_tac2_words = {
    "working_dir":  "/ch-ch-words",
    "model_file" : "./exp/tts_train+xvector_tacotron2_raw_word_use_wandbtrue_resumetrue_wandb_idtacotron2-ch-ch-word/100epoch.pth",
    "vocoder_file": melgan_vocoder_file
}

ch_ch_tac2_char = {
    "working_dir":  "/ch-ch",
    "model_file" : "./exp/tts_train+xvector_tacotron2_raw_char_use_wandbtrue_resumetrue_wandb_idtacotron2-ch-ch-char/110epoch.pth",
    "vocoder_file": melgan_vocoder_file
}

ch_vits_char_swissDial = {
    "working_dir":  "/ch-swissDial-vits",
    "model_file" : "./exp/22k/tts_train+xvector_vits_raw_char_use_wandbtrue_resumetrue_wandb_idvits-ch-ch-char-swissDial/531epoch.pth",
    "vocoder_file": ""
}

ch_mixed = {
    "working_dir": "/ch-mixed",
    "model_file": "./exp/tts_train+xvector_tacotron2_raw_char_use_wandbtrue_resumetrue_wandb_idtacotron2-ch-ch-char-mixed/223epoch.pth",
    "vocoder_file": melgan_vocoder_file
}

ch_slowsoft_phn = {
    "working_dir":  "/ch-slowsoft-phn",
    "model_file" : "./exp/22k/tts_train_vits_raw_phn_xampa_tokenizer_use_wandbtrue_resumetrue_wandb_idvits-ph-slowsoft/117epoch.pth",
    "vocoder_file": ""
}