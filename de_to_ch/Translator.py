import os

from de_to_ch.utils import setup_device

from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5 import T5Tokenizer


class Translator:
    def __init__(self, model_output_dir="de_to_ch/experiments/transcribed_version__20220721_104626"):
        self.model_output_dir = model_output_dir
        self.t5 = "t5-small"
        self.beam_size = 1
        self.device, n_gpu = setup_device()

        self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(self.model_output_dir, 'best-model'))
        self.tokenizer = T5Tokenizer.from_pretrained(self.t5)
        self.model.to(self.device)

    def translate(self, sentences):
        input_batch_pt = self.tokenizer(
            sentences,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        decoded_out = self.model.generate(
            input_batch_pt['input_ids'].to(self.device),
            max_length=64,
            pad_token_id=self.tokenizer.eos_token_id,
            num_beams=self.beam_size,
            num_return_sequences=1
        )
        pred_batch = self.tokenizer.batch_decode(decoded_out, skip_special_tokens=True)
        return pred_batch

    def translate_one(self, sentence: str):
        return self.translate([sentence])[0]
