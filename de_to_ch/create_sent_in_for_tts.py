
with open('data/snf_train_ch_zh.tsv', 'rt', encoding='utf-8') as ifile, open('data/snf_train_ch_zh_tts.txt', 'wt', encoding='utf-8') as ofile:
    for line in ifile:
        sline = line.split('\t')
        ofile.write(sline[1])