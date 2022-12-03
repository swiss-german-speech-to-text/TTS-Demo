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
global model_data
global translator
model_data = ch_vits_char_swissDial
translator = Translator("de_to_ch/experiments/transcribed_version__20220721_104626")

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


def interference_ch(speaker_id: int, text_ch: str, xvector: XVector = None) -> Tuple[np.ndarray, int]:
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
    wd = str(Path().absolute())
    model_dir = wd + model_data["working_dir"]
    # os.chdir(wd + model_data["working_dir"])

    if xvector is None:
        xvector = XVector(model_dir + "/dump/**/spk_xvector.ark")

    tts = Text2Speech.from_pretrained(
        model_file=os.path.join(model_dir, model_data["model_file"]),
        vocoder_file=os.path.join(model_dir, model_data["vocoder_file"]))

    spembs = xvector.get_spembs(speaker_id)

    with torch.no_grad():
        out = tts(text_ch, spembs=spembs)
        wav = out["wav"]
    wav_series = wav.view(-1).cpu().numpy()

    return wav_series, tts.fs


def translate_to_ch(text_de: str, dialect: int):
    dialect = id_to_dialect[dialect]
    return translator.translate_one(f"{dialect}: {text_de}")

@app.route("/")
def streamwav():
    text_de = request.args.get('text_de', '')
    text_ch = request.args.get('text_ch', '')
    dialect = request.args.get('dialect', None)
    translate = request.args.get('translate', None)
    synthesize = request.args.get('synthesize', None)
    audio_data = None

    if translate is not None and text_de != '' and dialect is not None:
        text_ch = translate_to_ch(text_de, int(dialect))

    if synthesize is not None and text_ch != '' and dialect is not None:
        wav, sr = interference_ch(int(dialect), text_ch)

        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, sr, wav)
        wav_bytes = byte_io.read()

        audio_data = base64.b64encode(wav_bytes).decode('UTF-8')
    return render_template("result.html", audio_data=audio_data, text_de=text_de, text_ch=text_ch, dialect=dialect)


@app.route('/wav2', methods=['POST'])
def streamwav3():
    text_de = request.form['text']
    wav, sr = interference_ch(0, text_de)

    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    write(byte_io, sr, wav)
    wav_bytes = byte_io.read()

    return Response(wav_bytes, mimetype="audio/x-wav")


if __name__ == '__main__':
    app.run()