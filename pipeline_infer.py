
import hook_proc
from tools.uvr5.webui import uvr
import os
from subprocess import Popen

from GPT_SoVITS.inference_webui import get_tts_wav

get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language)