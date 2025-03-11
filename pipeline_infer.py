
import hook_proc
import os
from subprocess import Popen
import argparse

from tools.uvr5.webui import uvr
from GPT_SoVITS.inference_webui import get_tts_wav

# ref_wav_path = 'ref.wav'
# ref_text = 'ref_text'
# text = 'text'

def get_tts(ref_wav_path, ref_text, text):
    for msg in get_tts_wav(
        ref_wav_path, 
        ref_text, 
        'Chinese',
        text, 
        'Chinese',
        '不切',
        15,
        1,
        1, 
        False,
        1,
        False, 
        None,
        32,
        False,
        0.3):
        print(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_wav_path', type=str, required=True)
    parser.add_argument('--ref_text', type=str, required=True)
    parser.add_argument('--text', type=str, required=True)
    args = parser.parse_args()
    ref_wav_path = args.ref_wav_path
    ref_text = args.ref_text
    text = args.text
    get_tts(ref_wav_path, ref_text, text)
    print('done')