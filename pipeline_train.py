
import hook_proc
from tools.uvr5.webui import uvr_ex
import os
from subprocess import Popen

from GPT_SoVITS.inference_webui import get_tts_wav

################################################################

vocal_dir = "output/uvr5_opt"
slicer_dir = "output/slicer_opt"
denoise_dir = "output/denoise_opt"
asr_dir = "output/asr_opt"
input_audio_dir = "data/input_audio"  #放到这个目录

################################################################

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
uvr_ex(
    model_name="HP2_all_vocals",  #fixme: change it
    inp_root=input_audio_dir, 
    save_root_vocal=vocal_dir,
    paths=[], 
    save_root_ins=vocal_dir,
    agg=10,
    format0="m4a",
    device_='cuda',
    is_half_=True,
)
# 输出到save_root_vocal和save_root_ins

# os.system('python tools/cmd-denoise.py -i "output/slicer_opt" -o "output/uvr5_opt"')


print(f'start slice to {slicer_dir}')
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 0 4', shell=True)
print(p)
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 1 4', shell=True)
print(p)
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 2 4', shell=True)
print(p)
p = Popen(f'python tools/slice_audio.py {vocal_dir} "{slicer_dir}" -34 4000 300 10 500 0.9 0.25 3 4', shell=True)
print(p)


print(f'start denoise to {denoise_dir}')
p = Popen(f'"python" tools/cmd-denoise.py -i "{slicer_dir}" -o "{denoise_dir}" -p float16', shell=True)
print(p)


print(f'start asr to {asr_dir}')
p = Popen(f'"python" tools/asr/funasr_asr.py -i "{denoise_dir}" -o "{asr_dir}" -s large -l zh -p float32', shell=True)
print(p)

'''
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
'''

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
