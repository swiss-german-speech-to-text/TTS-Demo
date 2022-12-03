import os,json

from data_loader import dialects
from config import read_arguments_train
from utils import setup_device
from data_collator import DataCollatorForSQL2Text

from datasets import load_metric

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from tqdm import tqdm

metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def translate_single_sentence(in_sent):
    input_batch_pt = tokenizer(
        [in_sent],
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding=True
    )
    decoded_out = model.generate(
        input_batch_pt['input_ids'].to(device),
        max_length=64,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=args.beam_size,
        num_return_sequences=1
    )
    pred_batch = tokenizer.batch_decode(decoded_out, skip_special_tokens=True)
    return pred_batch[0]

def compute_metrics():
    inputs, out_labels, out_preds = [], [], []

    n_decoding_steps = int(len(eval_df) / args.eval_batch_size) + 1
    for in_batch in tqdm(batch_list(eval_df, n=args.eval_batch_size), desc="Decoding:", total=n_decoding_steps):
        label_batch = [x[1] for x in in_batch]
        input_batch = [x[0] for x in in_batch]
        input_batch_pt = tokenizer(
            input_batch,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding=True
        )
        decoded_out = model.generate(
            input_batch_pt['input_ids'].to(device),
            max_length=64,
            pad_token_id=tokenizer.eos_token_id,
            num_beams=args.beam_size,
            num_return_sequences=1
        )
        pred_batch = tokenizer.batch_decode(decoded_out, skip_special_tokens=True)

        out_labels.extend(label_batch)
        out_preds.extend(pred_batch)
        inputs.extend(input_batch)

    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    result = {k: round(v, 4) for k, v in result.items()}

    with open(os.path.join(args.model_output_dir, f'results_final.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for inp, pred, label in zip(inputs, decoded_preds, decoded_labels):
            f.write(f"{inp}\t{pred}\t{label[0]}\n")
    return result


if __name__ == '__main__':
    args = read_arguments_train()
    device, n_gpu = setup_device()

    with open(os.path.join(args.model_output_dir, 'eval_df.json'), 'rt', encoding='utf-8') as ifile:
        eval_df = json.load(ifile)

    model = T5ForConditionalGeneration.from_pretrained(os.path.join(args.model_output_dir, 'best-model'))
    tokenizer = T5Tokenizer.from_pretrained(args.t5)
    data_collator = DataCollatorForSQL2Text(tokenizer, 128, device)
    model.to(device)
    if not args.toy:
        compute_metrics()
    else:
        while True:
            try:
                dial_prompt = '\t'.join([f'{i}:{dial}' for i, dial in enumerate(dialects)])
                selected_dialect = input(f'Select Dialect: {dial_prompt}')
                dial_tag = dialects[int(selected_dialect)]
                input_sentence = input("Enter German Sentence:")
                out_sentence = translate_single_sentence(f'[{dial_tag}]: {input_sentence}')
                print(out_sentence)
            except:
                pass