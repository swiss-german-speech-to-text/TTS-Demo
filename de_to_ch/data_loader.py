import json, random
import pandas as pd

dialects = [
    'ch_zh',
    'ch_sg',
    'ch_be',
    'ch_gr',
    'ch_vs',
    'ch_bs',
    'ch_ag',
    'ch_lu'
]


def convert_to_pandas(dataset, dataset_tr, ids, direction):
    data_points = []
    id_to_dp = {x['id']: x for x in dataset}
    id_to_dp_tr = {x['id']: x for x in dataset_tr}
    for data_id in ids:
        de_utterance = id_to_dp[data_id]['de']
        entry = id_to_dp_tr[data_id]
        for dialect in dialects:
            utt = entry.get(dialect)
            if utt is not None:
                if direction == 'de_to_ch':
                    dp = (f'[{dialect}]: {de_utterance}', utt)
                else:
                    dp = (utt, f'[{dialect}]: {de_utterance}')
                data_points.append(dp)
    return data_points


def load_data_raw(n_eval_pairs=500, direction='de_to_ch'):
    with open('data/sentences_ch_de_numerics.json', 'rt', encoding='utf-8') as ifile:
        data = json.load(ifile)
    if direction == 'de_to_ch':
        with open('data/sentences_ch_de_transcribed.json', 'rt', encoding='utf-8') as ifile:
            data_tr = json.load(ifile)
    else:
        data_tr = data
    print(f'Number of Datapoints: {len(data)}')
    full_dps = [x['id'] for x in data if len(x) == 11]
    print(f'Number of Datapoints with all dialects: {len(full_dps)}')
    n_pairs = sum([len(x) - 3 for x in data if len(x) == 11])
    print(f'Full Number of Pairs: {n_pairs}')
    random.shuffle(full_dps)
    eval_ids = set(full_dps[:n_eval_pairs])
    train_data_ids = [x['id'] for x in data if x['id'] not in eval_ids]
    eval_data_ids = [x['id'] for x in data if x['id'] in eval_ids]

    print(f'Train Set Size: {len(train_data_ids)} - Eval Set Size: {len(eval_data_ids)}')
    train_df = convert_to_pandas(data, data_tr, train_data_ids, direction=direction)
    eval_df = convert_to_pandas(data, data_tr, eval_data_ids, direction=direction)

    return train_df, eval_df
