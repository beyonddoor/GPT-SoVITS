
import hook_proc
from tools.uvr5.webui import uvr
import os
from subprocess import Popen

from GPT_SoVITS.inference_webui import get_tts_wav

ref_wav_path = 'ref.wav'
prompt_text = 'prompt_text'
text = 'text'

for msg in get_tts_wav(
    ref_wav_path, 
    prompt_text, 
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