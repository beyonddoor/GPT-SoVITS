import hook_proc

import os
from subprocess import Popen

print('import tools.uvr5.webui')
from tools.uvr5.webui import uvr_ex
import webui
from icecream import ic
log_debug = ic


def wait_proc(p):
    if p:
        print(f'wait proc {p}')
        p.wait()

################################################################

vocal_dir = "output/uvr5_opt"
instrument_dir = "output/uvr5_opt"
slicer_dir = "output/slicer_opt"
denoise_dir = "output/denoise_opt"
asr_dir = "output/asr_opt"
input_audio_dir = "data/input_audio"  #放到这个目录

################################################################

print('import finished')

if not os.path.exists(input_audio_dir):
    raise FileNotFoundError(f"input_audio_dir not found: {input_audio_dir}")

if not os.listdir(input_audio_dir):
    raise FileNotFoundError("input_audio_dir contains files")

# for v in uvr_ex(
#     model_name="HP2_all_vocals",  #fixme: change it
#     inp_root=input_audio_dir, 
#     save_root_vocal=vocal_dir,
#     paths=[], 
#     save_root_ins=vocal_dir,
#     agg=10,
#     format0="m4a",
#     device_='cuda',
#     is_half_=True,
# ):
#     print(v)

print('start uvr')
uvr_ex(
    model_name="HP2_all_vocals",  #fixme: change it
    inp_root='', 
    save_root_vocal=vocal_dir,
    paths=[f'{input_audio_dir}/1.m4a'], 
    save_root_ins=instrument_dir,
    agg=10,
    format0="m4a",
    device_='cuda',
    is_half_=True,
)
# 输出到save_root_vocal和save_root_ins

log_debug(vocal_dir, os.listdir(vocal_dir))
if len(os.listdir(vocal_dir)) == 0:
    raise FileNotFoundError("vocal_dir contains no files")


# os.system('python tools/cmd-denoise.py -i "output/slicer_opt" -o "output/uvr5_opt"')


print(f'start slice to {slicer_dir}')
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 0 4', shell=True)
wait_proc(p)
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 1 4', shell=True)
wait_proc(p)
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 2 4', shell=True)
wait_proc(p)
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 3 4', shell=True)
wait_proc(p)


print(f'start denoise to {denoise_dir}')
p = Popen(f'"python" tools/cmd-denoise.py -i "{slicer_dir}" -o "{denoise_dir}" -p float16', shell=True)
wait_proc(p)


print(f'start asr to {asr_dir}')
p = Popen(f'"python" tools/asr/funasr_asr.py -i "{denoise_dir}" -o "{asr_dir}" -s large -l zh -p float32', shell=True)
wait_proc(p)

print(os.getcwd())


# pass data via env vars

# config={
#     "inp_text":inp_text,
#     "inp_wav_dir":inp_wav_dir,
#     "exp_name":exp_name,
#     "opt_dir":opt_dir,
#     "bert_pretrained_dir":bert_pretrained_dir,
#     "is_half": str(is_half)
# }
# gpu_names=gpu_numbers1a.split("-")
# all_parts=len(gpu_names)
# for i_part in range(all_parts):
#     config.update(
#         {
#             "i_part": str(i_part),
#             "all_parts": str(all_parts),
#             "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
#         }
#     )
#     os.environ.update(config)

# p = Popen('python GPT_SoVITS/prepare_datasets/1-get-text.py', shell=True)
# wait_proc(p)

# p = Popen('python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', shell=True)
# wait_proc(p)

# p = Popen('python GPT_SoVITS/prepare_datasets/3-get-semantic.py', shell=True)
# wait_proc(p)

for msg in webui.open1abc(
    'output/asr_opt/denoise_opt.list', 
    'output/denoise_opt',
    'GPT-SoVITS',
    '0-0',
    '0-0',
    '0-0',
    'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
    'GPT_SoVITS/pretrained_models/chinese-hubert-base',
    'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth'
):
    print(msg)

# p = Popen('python GPT_SoVITS/s2_train.py --config "/content/GPT-SoVITS/TEMP/tmp_s2.json"', shell=True)
# wait_proc(p)

gpus = '0-0'

for msg in webui.open1Ba(
    7,8,'GPT-SoVITS',
    0.4, True, True,
    4, gpus, 
    'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth',
    'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth',
    False,
    32
):
    print(msg)

# p = Popen('python GPT_SoVITS/s1_train.py --config_file "/content/GPT-SoVITS/TEMP/tmp_s1.yaml"', shell=True)
# wait_proc(p)

for msg in webui.open1Bb(
    7, 15, 'GPT-SoVITS', 
    False,True, True, 5, gpus, 
    'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
):
    print(msg)

