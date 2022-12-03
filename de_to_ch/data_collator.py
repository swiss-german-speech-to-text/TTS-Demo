import torch, copy
from transformers import PreTrainedTokenizer


class DataCollatorForSQL2Text:
    label_pad_token_id: int = -100

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_len: int,
            device
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = device

    def __call__(self, batch, return_tensors=None):
        input = self.tokenizer(
            [x[0] for x in batch],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding=True
        )
        labels = self.tokenizer(
            [x[1] for x in batch],
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding=True
        )

        out_batch = {
            "input_ids": input['input_ids'],
            "attention_mask": input['attention_mask'],
            "labels": labels['input_ids'],
            "return_dict": True
        }

        return out_batch