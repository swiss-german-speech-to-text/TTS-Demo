import os
from pathlib import Path
import torch

from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
import ctranslate2


class Translator:
    def __init__(self, model_output_dir="de_to_ch/experiments/transcribed_version__20220721_104626"):
        self.model_output_dir = model_output_dir
        self.t5 = "t5-small"

        ct2_model_path = Path(self.model_output_dir, 'ct2-model')
        ct2_model_path.mkdir(parents=True, exist_ok=True)

        ct = ctranslate2.converters.TransformersConverter(model_name_or_path=os.path.join(self.model_output_dir, 'best-model'))
        ct.convert(output_dir=ct2_model_path, force=True)

        self.translator = ctranslate2.Translator(str(ct2_model_path), device="cpu", compute_type="auto")
        self.tokenizer = T5TokenizerFast.from_pretrained(self.t5)

    def translate_one(self, sentence: str):
        with torch.inference_mode():
            input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(sentence))

            results = self.translator.translate_batch([input_tokens])
            output_tokens = results[0].hypotheses[0]

            output_text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(output_tokens))

        return output_text
