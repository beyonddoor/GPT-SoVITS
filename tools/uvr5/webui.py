print(f'--> webui')

import os
import hook_proc

import traceback,gradio as gr
import logging
from tools.i18n.i18n import I18nAuto
from tools.my_utils import clean_path

# 语言自动
i18n = I18nAuto()

logger = logging.getLogger(__name__)
import librosa,ffmpeg
import soundfile as sf
import torch
import sys
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import Roformer_Loader

try:
    import gradio.analytics as analytics
    analytics.version_check = lambda:None
except:...

weight_uvr5_root = "tools/uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))

print(f'uvr5_names={uvr5_names}')
device=None
is_half=None

# 如下与webui有关
webui_port_uvr5=None
is_share=None

if __name__ == "__main__":
    device=sys.argv[1]
    is_half=eval(sys.argv[2])
    webui_port_uvr5=int(sys.argv[3])
    is_share=eval(sys.argv[4])

def html_left(text, label='p'):
    return f"""<div style="text-align: left; margin: 0; padding: 0;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""

def html_center(text, label='p'):
    return f"""<div style="text-align: center; margin: 100; padding: 50;">
                <{label} style="margin: 0; padding: 0;">{text}</{label}>
                </div>"""


# 提取主人声音
def uvr_new(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    '''
    model_name: str, 模型名。tools/uvr5/uvr5_weights下的模型名
    inp_root: str, 输入文件夹路径
    save_root_vocal: str, 输出主人声文件夹路径
    paths: list, 输入文件路径
    save_root_ins: str, 输出非主人声文件夹路径
    agg: str, 人声提取激进程度
    format0: str, 导出文件格式
    '''
    print(f'uvr {model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0}')

    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name

        # 三种模型的处理
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif "roformer" in model_name.lower():
            func = Roformer_Loader
            pre_fun = func(
                model_path = os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                config_path = os.path.join(weight_uvr5_root, model_name + ".yaml"),
                device = device,
                is_half=is_half
            )
            if not os.path.exists(os.path.join(weight_uvr5_root, model_name + ".yaml")):
                infos.append("Warning: You are using a model without a configuration file. The program will automatically use the default configuration file. However, the default configuration file cannot guarantee that all models will run successfully. You can manually place the model configuration file into 'tools/uvr5/uvr5w_weights' and ensure that the configuration file is named as '<model_name>.yaml' then try it again. (For example, the configuration file corresponding to the model 'bs_roformer_ep_368_sdr_12.9628.ckpt' should be 'bs_roformer_ep_368_sdr_12.9628.yaml'.) Or you can just ignore this warning.")
                print("\n".join(infos))
                return
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )

        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            # 上传的文件列表
            paths = [path.name for path in paths]

        print(f'uvr paths={paths}')

        for path in paths:
            inp_path = path
            print(f'uvr processing inp_path={inp_path}')
            if(os.path.isfile(inp_path)==False):continue
            need_reformat = 1
            done = 0
            try:
                print('uvr step 1')
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    print(f"No need to reformat {inp_path}")
                    print(f"==> Start to process {inp_path}")
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()

            
            print('uvr step 2')
            # 处理音频格式
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                print(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
                print('uvr step 3')
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path

            print('uvr step 4')
            try:
                # 转换后再处理一下
                print('uvr step 5')
                if done == 0:
                    print(f"==> Start to process {inp_path}")
                    print('uvr step 6')
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                print("\n".join(infos))
                return
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                print("\n".join(infos))
                return
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        infos.append(traceback.format_exc())
        print("\n".join(infos))
        return
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()

        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n".join(infos))
    return

# 提取主人声音
def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    '''
    model_name: str, 模型名。tools/uvr5/uvr5_weights下的模型名
    inp_root: str, 输入文件夹路径
    save_root_vocal: str, 输出主人声文件夹路径
    paths: list, 输入文件路径
    save_root_ins: str, 输出非主人声文件夹路径
    agg: str, 人声提取激进程度
    format0: str, 导出文件格式
    '''
    print(f'uvr {model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0}')

    infos = []
    try:
        inp_root = clean_path(inp_root)
        save_root_vocal = clean_path(save_root_vocal)
        save_root_ins = clean_path(save_root_ins)
        is_hp3 = "HP3" in model_name

        # 三种模型的处理
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        elif "roformer" in model_name.lower():
            func = Roformer_Loader
            pre_fun = func(
                model_path = os.path.join(weight_uvr5_root, model_name + ".ckpt"),
                config_path = os.path.join(weight_uvr5_root, model_name + ".yaml"),
                device = device,
                is_half=is_half
            )
            if not os.path.exists(os.path.join(weight_uvr5_root, model_name + ".yaml")):
                infos.append("Warning: You are using a model without a configuration file. The program will automatically use the default configuration file. However, the default configuration file cannot guarantee that all models will run successfully. You can manually place the model configuration file into 'tools/uvr5/uvr5w_weights' and ensure that the configuration file is named as '<model_name>.yaml' then try it again. (For example, the configuration file corresponding to the model 'bs_roformer_ep_368_sdr_12.9628.ckpt' should be 'bs_roformer_ep_368_sdr_12.9628.yaml'.) Or you can just ignore this warning.")
                yield "\n".join(infos)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )

        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            # 上传的文件列表
            paths = [path.name for path in paths]

        print(f'uvr paths={paths}')

        for path in paths:
            inp_path = os.path.join(inp_root, path)
            print(f'uvr processing inp_path={inp_path}')
            if(os.path.isfile(inp_path)==False):continue
            need_reformat = 1
            done = 0
            try:
                print('uvr step 1')
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                    info["streams"][0]["channels"] == 2
                    and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    print(f"No need to reformat {inp_path}")
                    print(f"==> Start to process {inp_path}")
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()

            
            print('uvr step 2')
            # 处理音频格式
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                print(f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y')
                print('uvr step 3')
                os.system(
                    f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
                )
                inp_path = tmp_path

            print('uvr step 4')
            try:
                # 转换后再处理一下
                print('uvr step 5')
                if done == 0:
                    print(f"==> Start to process {inp_path}")
                    print('uvr step 6')
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0,is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()

        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)

def uvr_ex(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, device_, is_half_):
    print(f'uvr_ex {model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0, device_, is_half_}')
    global device,is_half
    device=device_
    is_half=is_half_
    print('start to uvr')
    res = uvr_new(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0)
    print(f'uvr_ex done', res)
    return res

if __name__ == "__main__":

    print(f'uvr5 webui start')

    with gr.Blocks(title="UVR5 WebUI") as app:
        gr.Markdown(
            value=
                i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责.") + "<br>" + i18n("如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录LICENSE.")
        )
        with gr.Group():
            gr.Markdown(html_center(i18n("伴奏人声分离&去混响&去回声"),'h2'))
            with gr.Group():
                    gr.Markdown(
                        value=html_left(i18n("人声伴奏分离批量处理， 使用UVR5模型。") + "<br>" + \
                            i18n("合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。")+ "<br>" + \
                            i18n("模型分为三类：") + "<br>" + \
                            i18n("1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点；") + "<br>" + \
                            i18n("2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型；") + "<br>" + \
                            i18n("3、去混响、去延迟模型（by FoxJoy）：") + "<br>  " + \
                            i18n("(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；") + "<br>&emsp;" + \
                            i18n("(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。") + "<br>" + \
                            i18n("去混响/去延迟，附：") + "<br>" + \
                            i18n("1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；") + "<br>" + \
                            i18n("2、MDX-Net-Dereverb模型挺慢的；") + "<br>" + \
                            i18n("3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"),'h4')
                    )
                    with gr.Row():
                        with gr.Column():
                            model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
                            dir_wav_input = gr.Textbox(
                                label=i18n("输入待处理音频文件夹路径"),
                                placeholder="C:\\Users\\Desktop\\todo-songs",
                            )

                            # 通过文件上传的audio
                            wav_inputs = gr.File(
                                file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                            )
                        with gr.Column():
                            agg = gr.Slider(
                                minimum=0,
                                maximum=20,
                                step=1,
                                label=i18n("人声提取激进程度"),
                                value=10,
                                interactive=True,
                                visible=False,  # 先不开放调整
                            )
                            opt_vocal_root = gr.Textbox(
                                label=i18n("指定输出主人声文件夹"), value="output/uvr5_opt"
                            )
                            opt_ins_root = gr.Textbox(
                                label=i18n("指定输出非主人声文件夹"), value="output/uvr5_opt"
                            )
                            format0 = gr.Radio(
                                label=i18n("导出文件格式"),
                                choices=["wav", "flac", "mp3", "m4a"],
                                value="flac",
                                interactive=True,
                            )
                            with gr.Column():
                                with gr.Row():
                                    but2 = gr.Button(i18n("转换"), variant="primary")
                                with gr.Row():
                                    vc_output4 = gr.Textbox(label=i18n("输出信息"),lines=3)
                        but2.click(
                            uvr,
                            [
                                model_choose,
                                dir_wav_input,
                                opt_vocal_root,
                                wav_inputs,
                                opt_ins_root,
                                agg,
                                format0,
                            ],
                            [vc_output4],
                            api_name="uvr_convert",
                        )
                        
    app.queue().launch(#concurrency_count=511, max_size=1022
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=webui_port_uvr5,
        quiet=True,
    )


print(f'webui import done')