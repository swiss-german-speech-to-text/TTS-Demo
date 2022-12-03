import os,json

from data_loader import load_data_raw
from config import read_arguments_train
from utils import setup_device, create_experiment_folder
from data_collator import DataCollatorForSQL2Text

from datasets import load_metric

from transformers import SchedulerType
from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer
from tqdm import tqdm

metric = load_metric("sacrebleu")
best_bleu = 0.0

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


n_output = 1


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def compute_metrics(eval_preds):
    global n_output
    global best_bleu
    out_labels, out_preds = [], []

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
        decoded_out = model.generate(input_batch_pt['input_ids'].to(device), max_length=64, pad_token_id=tokenizer.eos_token_id)
        pred_batch = tokenizer.batch_decode(decoded_out, skip_special_tokens=True)

        out_labels.extend(label_batch)
        out_preds.extend(pred_batch)

    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    result = {k: round(v, 4) for k, v in result.items()}
    out_n = int(n_output * eval_steps)

    if result["bleu"] > best_bleu:
        model.save_pretrained(os.path.join(output_path, 'best-model'), save_config=True)
        best_bleu = result["bleu"]

    with open(os.path.join(output_path, f'results_{out_n}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")
    n_output += 1
    return result


if __name__ == '__main__':
    args = read_arguments_train()
    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)

    device, n_gpu = setup_device()

    train_df, eval_df = load_data_raw(n_eval_pairs=250, direction=args.direction)
    if args.toy:
        train_df = train_df[:20]
        eval_df = eval_df[:20]
    with open(os.path.join(output_path, 'train_df.json'), 'wt', encoding='utf-8') as ofile:
        json.dump(train_df, ofile)
    with open(os.path.join(output_path, 'eval_df.json'), 'wt', encoding='utf-8') as ofile:
        json.dump(eval_df, ofile)

    # load (supports t5, mt5, byT5 models)
    model = T5ForConditionalGeneration.from_pretrained(args.t5)
    tokenizer = T5Tokenizer.from_pretrained(args.t5)
    data_collator = DataCollatorForSQL2Text(tokenizer, 128, device)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    perc = 100 * (pytorch_trainable_params / pytorch_total_params)
    print(f'Training {pytorch_trainable_params} out of {pytorch_total_params} parameters ({perc})!')

    # track the model
    # wandb.watch(model, log='parameters')
    eval_steps = 1
    nocuda = not str(device) == 'cuda'
    train_args = TrainingArguments(
        output_dir=output_path,
        logging_dir=output_path,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        save_total_limit=2,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_every_n_steps,
        eval_accumulation_steps=args.eval_acc,
        no_cuda=nocuda,
        fp16=args.fp16,
        bf16=args.bf16,
        bf16_full_eval=args.bf16,
        save_strategy="epoch",
        ignore_data_skip=True,
        logging_steps=10,
        learning_rate=args.lr_base,
        warmup_steps=100,
        dataloader_pin_memory=False,
        lr_scheduler_type=SchedulerType.LINEAR,
        label_smoothing_factor=0.0
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_df,
        eval_dataset=eval_df,
        compute_metrics=compute_metrics
    )

    ignore_keys_for_eval = ['past_key_values', 'encoder_last_hidden_state', 'hidden_states', 'cross_attentions',
                            'logits']
    trainer.train(ignore_keys_for_eval=ignore_keys_for_eval)
