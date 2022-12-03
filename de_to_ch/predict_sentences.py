import os,csv

import torch

from data_loader import dialects
from config import read_arguments_train
from utils import setup_device
from data_collator import DataCollatorForSQL2Text

from datasets import load_metric

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from tqdm import tqdm



def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def compute_metrics(dialect_label):
    inputs, out_preds = [], []

    n_decoding_steps = int(len(eval_df) / args.eval_batch_size) + 1
    for in_batch in tqdm(batch_list(eval_df, n=args.eval_batch_size), desc="Decoding:", total=n_decoding_steps):
        input_batch = [f'[{dialect_label}]: {x}' for x in in_batch]
        input_batch_pt = tokenizer(
            input_batch,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        with torch.no_grad():
            decoded_out = model.generate(
                input_batch_pt['input_ids'].to(device),
                max_length=64,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=args.beam_size,
                num_return_sequences=1
            )
        pred_batch = tokenizer.batch_decode(decoded_out, skip_special_tokens=True)

        out_preds.extend(pred_batch)
        inputs.extend(input_batch)

    return inputs, out_preds


if __name__ == '__main__':
    args = read_arguments_train()
    device, n_gpu = setup_device()

    eval_df = []
    with open(os.path.join('data', 'snf_train_v_TM28k_AA4k_7x.csv'), 'rt', encoding='utf-8') as ifile:
        reader = csv.reader(ifile, delimiter=';')
        for row in reader:
            eval_df.append(row[-1])
    if args.toy:
        eval_df = eval_df[:200]

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(args.model_output_dir, 'best-model'))
    tokenizer = T5Tokenizer.from_pretrained(args.t5)
    data_collator = DataCollatorForSQL2Text(tokenizer, 128, device)
    model.to(device)

    for dialect_label in dialects:
        inputs, out_preds = compute_metrics(dialect_label)
        with open(os.path.join('data', f'snf_train_{dialect_label}.tsv'), 'wt', encoding='utf-8') as ofile:
            for inp, pred in zip(inputs, out_preds):
                ofile.write(f'{inp}\t{pred}\n')

