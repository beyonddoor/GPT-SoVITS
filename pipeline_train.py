import hook_proc

import os
from subprocess import Popen
import argparse
from config import infer_device

################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-id', '--user_id', type=str, required=True)
parser.add_argument('--seperate', type=int, default=1)
args = parser.parse_args()

USE_USER_SPACE = args.seperate == 1

if USE_USER_SPACE:
    user_id = args.user_id
    input_audio_dir = f"data/{user_id}/input_audio"  #放到这个目录
    output_prefix = f'output/{user_id}'
else:
    user_id = None
    input_audio_dir = f"data/1/input_audio"  #放到这个目录
    output_prefix = f'output/'
    

input_format = 'mp3'
gpus = '0-0'
################################################################

if USE_USER_SPACE:
    # 传递上下文，需要尽早执行
    os.environ['SOVITS_USER_ID'] = user_id

print('import tools.uvr5.webui')

from tools.uvr5.webui import uvr_ex
import webui
from icecream import ic

vocal_dir = f"{output_prefix}/uvr5_opt"
instrument_dir = f"{output_prefix}/uvr5_opt"
slicer_dir = f"{output_prefix}/slicer_opt"
denoise_dir = f"{output_prefix}/denoise_opt"
asr_dir = f"{output_prefix}/asr_opt"

print('import finished')

def wait_proc(p):
    if p:
        print(f'wait proc {p}')
        p.wait()

    if p.returncode != 0:
        raise Exception(f'proc failed: {p.returncode}')


def check_input_dir():
    if not os.path.exists(input_audio_dir):
        raise FileNotFoundError(f"input_audio_dir not found: {input_audio_dir}")

    dirs = os.listdir(input_audio_dir)
    if not dirs:
        raise FileNotFoundError("input_audio_dir contains files")

def uvr_data():
    print('start uvr')
    os.makedirs(output_prefix, exist_ok=True)

    uvr_ex(
        model_name="HP2_all_vocals",  #fixme: change it
        inp_root='', 
        save_root_vocal=vocal_dir,
        paths = [f'{input_audio_dir}/{name}' 
                for name in os.listdir(input_audio_dir) 
                if name.endswith(input_format) or name.endswith('.m4a')], 
        save_root_ins=instrument_dir,
        agg=10,
        format0=input_format,
        device_=infer_device,
        is_half_=True,
    )

    dirs = os.listdir(vocal_dir)
    ic(vocal_dir, dirs)
    if len(dirs) == 0:
        raise FileNotFoundError(f"{vocal_dir} contains no files")
    
    if len([d for d in dirs if d.startswith('vocal_')]) == 0:
        raise FileNotFoundError(f"{vocal_dir} contains no vocal files")

def prepare_data():
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

def train_data():
    print('start train SoVITS')
    for msg in webui.open1abc(
        f'{asr_dir}/denoise_opt.list', 
        f'{denoise_dir}',
        'GPT-SoVITS',
        gpus, gpus, gpus,
        'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large',
        'GPT_SoVITS/pretrained_models/chinese-hubert-base',
        'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth'
    ):
        print(msg)

    print('start open1Ba')
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

    print('start open1Bb')
    for msg in webui.open1Bb(
        7, 15, 'GPT-SoVITS', 
        False,True, True, 5, gpus, 
        'GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
    ):
        print(msg)


if __name__ == '__main__':
    check_input_dir()
    uvr_data()
    prepare_data()
    train_data()
    print('done')