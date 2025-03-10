import hook_proc
import os,sys
if len(sys.argv)==1:sys.argv.append('v2')
version="v1"if sys.argv[1]=="v1" else"v2"
os.environ["version"]=version
now_dir = os.getcwd()
sys.path.insert(0, now_dir)
import warnings
warnings.filterwarnings("ignore")
import json,yaml,torch,pdb,re,shutil
import platform
import psutil
import signal
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
torch.manual_seed(233333)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp
if(os.path.exists(tmp)):
    for name in os.listdir(tmp):
        if(name=="jieba.cache"):continue
        path="%s/%s"%(tmp,name)
        delete=os.remove if os.path.isfile(path) else shutil.rmtree
        try:
            delete(path)
        except Exception as e:
            print(str(e))
            pass
import site
import traceback
site_packages_roots = []
for path in site.getsitepackages():
    if "packages" in path:
        site_packages_roots.append(path)
if(site_packages_roots==[]):site_packages_roots=["%s/runtime/Lib/site-packages" % now_dir]
#os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
os.environ["all_proxy"] = ""
for site_packages_root in site_packages_roots:
    if os.path.exists(site_packages_root):
        try:
            with open("%s/users.pth" % (site_packages_root), "w") as f:
                f.write(
                    # "%s\n%s/runtime\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    "%s\n%s/GPT_SoVITS/BigVGAN\n%s/tools\n%s/tools/asr\n%s/GPT_SoVITS\n%s/tools/uvr5"
                    % (now_dir, now_dir, now_dir, now_dir, now_dir, now_dir)
                )
            break
        except PermissionError as e:
            traceback.print_exc()
from tools import my_utils
import shutil
import pdb
import subprocess
from subprocess import Popen

import signal
from config import python_exec,infer_device,is_half,exp_root,webui_port_main,webui_port_infer_tts,webui_port_uvr5,webui_port_subfix,is_share
from tools.i18n.i18n import I18nAuto, scan_language_list
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
os.environ["language"]=language
i18n = I18nAuto(language=language)
from scipy.io import wavfile
from tools.my_utils import load_audio, check_for_existance, check_details
from multiprocessing import cpu_count
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # 当遇到mps不支持的步骤时使用cpu
try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...
import gradio as gr
n_cpu=cpu_count()

ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

# 判断是否有能用来训练和加速推理的N卡
ok_gpu_keywords={"10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060","H","600","506","507","508","509"}
set_gpu_numbers=set()
if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ok_gpu_keywords):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            set_gpu_numbers.add(i)
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))

            
# # 判断是否支持mps加速
# if torch.backends.mps.is_available():
#     if_gpu_ok = True
#     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
#     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

def set_default():
    global default_batch_size,default_max_batch_size,gpu_info,default_sovits_epoch,default_sovits_save_every_epoch,max_sovits_epoch,max_sovits_save_every_epoch,default_batch_size_s1,if_force_ckpt
    if_force_ckpt = False
    if if_gpu_ok and len(gpu_infos) > 0:
        gpu_info = "\n".join(gpu_infos)
        minmem = min(mem)
        # if version == "v3" and minmem < 14:
        #     # API读取不到共享显存,直接填充确认
        #     try:
        #         torch.zeros((1024,1024,1024,14),dtype=torch.int8,device="cuda")
        #         torch.cuda.empty_cache()
        #         minmem = 14
        #     except RuntimeError as _:
        #         # 强制梯度检查只需要12G显存
        #         if minmem >= 12 :
        #             if_force_ckpt = True
        #             minmem = 14
        #         else:
        #             try:
        #                 torch.zeros((1024,1024,1024,12),dtype=torch.int8,device="cuda")
        #                 torch.cuda.empty_cache()
        #                 if_force_ckpt = True
        #                 minmem = 14
        #             except RuntimeError as _:
        #                 print("显存不足以开启V3训练")
        default_batch_size = minmem // 2 if version!="v3"else minmem//8
        default_batch_size_s1=minmem // 2
    else:
        gpu_info = ("%s\t%s" % ("0", "CPU"))
        gpu_infos.append("%s\t%s" % ("0", "CPU"))
        set_gpu_numbers.add(0)
        default_batch_size = default_batch_size_s1 = int(psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 4)
    if version!="v3":
        default_sovits_epoch=8
        default_sovits_save_every_epoch=4
        max_sovits_epoch=25#40
        max_sovits_save_every_epoch=25#10
    else:
        default_sovits_epoch=2
        default_sovits_save_every_epoch=1
        max_sovits_epoch=3#40
        max_sovits_save_every_epoch=3#10

    default_batch_size = max(1, default_batch_size)
    default_batch_size_s1 = max(1, default_batch_size_s1)
    default_max_batch_size = default_batch_size * 3

set_default()

gpus = "-".join([i[0] for i in gpu_infos])
default_gpu_numbers=str(sorted(list(set_gpu_numbers))[0])
def fix_gpu_number(input):#将越界的number强制改到界内
    try:
        if(int(input)not in set_gpu_numbers):return default_gpu_numbers
    except:return input
    return input
def fix_gpu_numbers(inputs):
    output=[]
    try:
        for input in inputs.split(","):output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs

pretrained_sovits_name=["GPT_SoVITS/pretrained_models/s2G488k.pth", "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth","GPT_SoVITS/pretrained_models/s2Gv3.pth"]
pretrained_gpt_name=["GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt","GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt", "GPT_SoVITS/pretrained_models/s1v3.ckpt"]

pretrained_model_list = (pretrained_sovits_name[int(version[-1])-1],pretrained_sovits_name[int(version[-1])-1].replace("s2G","s2D"),pretrained_gpt_name[int(version[-1])-1],"GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large","GPT_SoVITS/pretrained_models/chinese-hubert-base")

_ = ''
for i in pretrained_model_list:
    if "s2Dv3" not in i and os.path.exists(i) == False:
        _ += f'\n    {i}'
if _:
    print("warning: ", i18n('以下模型不存在:') + _)

_ = [[],[]]
for i in range(3):
    if os.path.exists(pretrained_gpt_name[i]):_[0].append(pretrained_gpt_name[i])
    else:_[0].append("")##没有下pretrained模型的，说不定他们是想自己从零训底模呢
    if os.path.exists(pretrained_sovits_name[i]):_[-1].append(pretrained_sovits_name[i])
    else:_[-1].append("")
pretrained_gpt_name,pretrained_sovits_name = _

SoVITS_weight_root=["SoVITS_weights","SoVITS_weights_v2","SoVITS_weights_v3"]
GPT_weight_root=["GPT_weights","GPT_weights_v2","GPT_weights_v3"]
for root in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(root,exist_ok=True)
def get_weights_names():
    SoVITS_names = [name for name in pretrained_sovits_name if name!=""]
    for path in SoVITS_weight_root:
        for name in os.listdir(path):
            if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (path, name))
    GPT_names = [name for name in pretrained_gpt_name if name!=""]
    for path in GPT_weight_root:
        for name in os.listdir(path):
            if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (path, name))
    return SoVITS_names, GPT_names

SoVITS_names,GPT_names = get_weights_names()
for path in SoVITS_weight_root+GPT_weight_root:
    os.makedirs(path,exist_ok=True)

def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names,key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names,key=custom_sort_key), "__type__": "update"}

p_label=None
p_uvr5=None
p_asr=None
p_denoise=None
p_tts_inference=None

# ------------------------------------ taken from webui.py ------------------------------------


import hook_proc
from tools.uvr5.webui import uvr_ex
import os
from subprocess import Popen

from GPT_SoVITS.inference_webui import get_tts_wav

def wait_proc(p):
    if p:
        print(f'wait proc {p}')
        p.wait()

################################################################

vocal_dir = "output/uvr5_opt"
instrument_dir = "output/uvr5_ins"
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
    save_root_ins=instrument_dir,
    agg=10,
    format0="m4a",
    device_='cuda',
    is_half_=True,
)
# 输出到save_root_vocal和save_root_ins

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

p = Popen('python GPT_SoVITS/prepare_datasets/1-get-text.py', shell=True)
wait_proc(p)

p = Popen('python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', shell=True)
wait_proc(p)

p = Popen('python GPT_SoVITS/prepare_datasets/3-get-semantic.py', shell=True)
wait_proc(p)

p = Popen('python GPT_SoVITS/s2_train.py --config "/content/GPT-SoVITS/TEMP/tmp_s2.json"', shell=True)
wait_proc(p)

p = Popen('python GPT_SoVITS/s1_train.py --config_file "/content/GPT-SoVITS/TEMP/tmp_s1.yaml"', shell=True)
wait_proc(p)
