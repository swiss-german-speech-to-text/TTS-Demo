import base64

from flask import Flask, Response, render_template, request
from typing import List, Tuple

import numpy as np
import torch
from espnet2.bin.tts_inference import Text2Speech

from tools.xvector import XVector

from scipy.io.wavfile import write
import os
from de_to_ch.Translator import Translator
from pathlib import Path
from tts_models import *
import io

app = Flask(__name__)

MAX_TEXT_LEN = 256

global model_data
global translator
global xvector
global tts
model_data = ch_vits_char_swissDial
translator = Translator("de_to_ch/experiments/transcribed_version__20220721_104626")
wd = str(Path().absolute())
model_dir = os.path.join(wd, model_data["working_dir"])
xvector = XVector(os.path.join(model_dir, "dump/**/spk_xvector.ark"))
tts = Text2Speech.from_pretrained(
    model_file=os.path.join(model_dir, model_data["model_file"]),
    vocoder_file=os.path.join(model_dir, model_data["vocoder_file"]))

id_to_dialect = {
    0: 'ag',
    1: 'be',
    2: 'bs',
    3: 'gr',
    4: 'lu',
    5: 'sg',
    6: 'vs',
    7: 'zh',
}


def inference_ch(speaker_id: int, text_ch: str) -> Tuple[np.ndarray, int]:
    """
    :param speaker_id:
    for swissDial:
    [(0, 'ag'),
     (1, 'be'),
     (2, 'bs'),
     (3, 'gr'),
     (4, 'lu'),
     (5, 'sg'),
     (6, 'vs'),
     (7, 'zh')]

    :param text_ch: german text.
    :param xvector: .
    :return: (wav series, sampling_rate)
    """
    spembs = xvector.get_spembs(speaker_id)

    with torch.no_grad():
        out = tts(text_ch, spembs=spembs)
        wav = out["wav"]
    wav_series = wav.view(-1).cpu().numpy()

    return wav_series, tts.fs


def translate_to_ch(text_de: str, dialect: int):
    dialect = id_to_dialect[dialect]
    return translator.translate_one(f"{dialect}: {text_de}")


@app.route("/tts", methods=['GET'])
def home():
    text_de = request.form.get('text_de', '')
    text_ch = request.form.get('text_ch', '')
    dialect = request.form.get('dialect', None)
    audio_data = None
    return render_template("result.html", audio_data=audio_data, text_de=text_de, text_ch=text_ch, dialect=dialect)


@app.route("/tts/translate", methods=['POST'])
def translate():
    text_de = request.form.get('text_de', '')
    text_ch = request.form.get('text_ch', '')
    dialect = request.form.get('dialect', None)
    audio_data = None

    if text_de != '' and len(text_de) <= MAX_TEXT_LEN and dialect is not None:
        text_de = " ".join(text_de.strip().split())
        text_de = text_de[0].upper() + text_de[1:] # Capitalize first letter
        # add a dot at the end if there is no punctuation
        if text_de[-1] not in [".", "!", "?"]:
            text_de += "."
        text_ch = translate_to_ch(text_de, int(dialect))
    return render_template("result.html", audio_data=audio_data, text_de=text_de, text_ch=text_ch, dialect=dialect)


@app.route("/tts/api_translate", methods=['POST'])
def translate():
    text_de = request.form.get('text_de', '')
    dialect = request.form.get('dialect', None)

    if text_de != '' and len(text_de) <= MAX_TEXT_LEN and dialect is not None:
        text_de = " ".join(text_de.strip().split())
        text_de = text_de[0].upper() + text_de[1:] # Capitalize first letter
        # add a dot at the end if there is no punctuation
        if text_de[-1] not in [".", "!", "?"]:
            text_de += "."
        text_ch = translate_to_ch(text_de, int(dialect))
    # return text_ch as json
    response = {'text_ch': text_ch}
    return jsonify(response)


@app.route("/tts/synthesize", methods=['POST'])
def synthesize():
    text_de = request.form.get('text_de', '')
    text_ch = request.form.get('text_ch', '')
    dialect = request.form.get('dialect', None)
    audio_data = None

    if text_ch != '' and len(text_ch) <= MAX_TEXT_LEN and dialect is not None:
        wav, sr = inference_ch(int(dialect), text_ch)

        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, sr, wav)
        wav_bytes = byte_io.read()

        audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
    return render_template("result.html", audio_data=audio_data, text_de=text_de, text_ch=text_ch, dialect=dialect)

@app.route("/tts/api_synthesize", methods=['POST'])
def synthesize():
    text_ch = request.form.get('text_ch', '')
    dialect = request.form.get('dialect', None)

    if text_ch != '' and len(text_ch) <= MAX_TEXT_LEN and dialect is not None:
        wav, sr = inference_ch(int(dialect), text_ch)

        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, sr, wav)
        wav_bytes = byte_io.read()

        audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
    response = {'audio_data': audio_data}
    return jsonify(response)


if __name__ == '__main__':
    app.run()
