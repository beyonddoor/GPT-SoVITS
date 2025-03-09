
import hook_proc
from tools.uvr5.webui import uvr
import os
from subprocess import Popen

from GPT_SoVITS.inference_webui import get_tts_wav

uvr(
    model_name="onnx_dereverb_By_FoxJoy",
    inp_root="tools/uvr5/test_data",
    save_root_vocal="tools/uvr5/test_data",
    paths=["tools/uvr5/test_data/1.wav"],
    save_root_ins="tools/uvr5/test_data",
    agg="mean",
    format0="wav"
)

# os.system('python tools/cmd-denoise.py -i "output/slicer_opt" -o "output/uvr5_opt"')

print('start slice')
p = Popen('python tools/slice_audio.py "output/uvr5_opt" "output/slicer_opt" -34 4000 300 10 500 0.9 0.25 0 4', shell=True)
print(p)
p = Popen('python tools/slice_audio.py "output/uvr5_opt" "output/slicer_opt" -34 4000 300 10 500 0.9 0.25 1 4', shell=True)
print(p)
p = Popen('python tools/slice_audio.py "output/uvr5_opt" "output/slicer_opt" -34 4000 300 10 500 0.9 0.25 2 4', shell=True)
print(p)
p = Popen('python tools/slice_audio.py "output/uvr5_opt" "output/slicer_opt" -34 4000 300 10 500 0.9 0.25 3 4', shell=True)
print(p)

print('start denoise')
p = Popen('"python" tools/cmd-denoise.py -i "output/slicer_opt" -o "output/denoise_opt" -p float16', shell=True)
print(p)

print('start asr')
p = Popen('"python" tools/asr/funasr_asr.py -i "output/denoise_opt" -o "output/asr_opt" -s large -l zh -p float32', shell=True)
print(p)

p = Popen('python GPT_SoVITS/prepare_datasets/1-get-text.py')
print(p)

p = Popen('python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py')
print(p)

p = Popen('python GPT_SoVITS/prepare_datasets/3-get-semantic.py')
print(p)

p = Popen('python GPT_SoVITS/s2_train.py --config "/content/GPT-SoVITS/TEMP/tmp_s2.json"')
print(p)

p = Popen('python GPT_SoVITS/s1_train.py --config_file "/content/GPT-SoVITS/TEMP/tmp_s1.yaml"')
print(p)
