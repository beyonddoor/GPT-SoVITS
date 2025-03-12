
# import hook_proc
# import os
# from subprocess import Popen
# import argparse
# from icecream import ic

# # from GPT_SoVITS.inference_webui import get_tts_wav
# # from api import get_tts_wav as get_tts_wav2

# from time import time as ttime
# import torch, torchaudio
# import numpy as np
# import librosa

# splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

# is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()

# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

# ssl_model = cnhubert.get_model()
# if is_half == True:
#     ssl_model = ssl_model.half().to(device)
# else:
#     ssl_model = ssl_model.to(device)

# from feature_extractor import cnhubert

# cnhubert.cnhubert_base_path = cnhubert_base_path

# def get_spepc(hps, filename):
#     # audio = load_audio(filename, int(hps.data.sampling_rate))
#     audio, sampling_rate = librosa.load(filename, sr=int(hps.data.sampling_rate))
#     audio = torch.FloatTensor(audio)
#     maxx=audio.abs().max()
#     if(maxx>1):audio/=min(2,maxx)
#     audio_norm = audio
#     audio_norm = audio_norm.unsqueeze(0)
#     spec = spectrogram_torch(
#         audio_norm,
#         hps.data.filter_length,
#         hps.data.sampling_rate,
#         hps.data.hop_length,
#         hps.data.win_length,
#         center=False,
#     )
#     return spec

# def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, top_k= 15, top_p = 0.6
#                 , temperature = 0.6, speed = 1, inp_refs = None
#                 , sample_steps = 32, if_sr = False, spk = "default"):
#     infer_sovits = speaker_list[spk].sovits
#     vq_model = infer_sovits.vq_model
#     hps = infer_sovits.hps
#     version = vq_model.version

#     infer_gpt = speaker_list[spk].gpt
#     t2s_model = infer_gpt.t2s_model
#     max_sec = infer_gpt.max_sec

#     t0 = ttime()
#     prompt_text = prompt_text.strip("\n")
#     if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
#     prompt_language, text = prompt_language, text.strip("\n")
#     dtype = torch.float16 if is_half == True else torch.float32
#     zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half == True else np.float32)
#     with torch.no_grad():
#         wav16k, sr = librosa.load(ref_wav_path, sr=16000)
#         wav16k = torch.from_numpy(wav16k)
#         zero_wav_torch = torch.from_numpy(zero_wav)
#         if (is_half == True):
#             wav16k = wav16k.half().to(device)
#             zero_wav_torch = zero_wav_torch.half().to(device)
#         else:
#             wav16k = wav16k.to(device)
#             zero_wav_torch = zero_wav_torch.to(device)
#         wav16k = torch.cat([wav16k, zero_wav_torch])
#         ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()
#         codes = vq_model.extract_latent(ssl_content)
#         prompt_semantic = codes[0, 0]
#         prompt = prompt_semantic.unsqueeze(0).to(device)

#         if version != "v3":
#             refers=[]
#             if(inp_refs):
#                 for path in inp_refs:
#                     try:
#                         refer = get_spepc(hps, path).to(dtype).to(device)
#                         refers.append(refer)
#                     except Exception as e:
#                         logger.error(e)
#             if(len(refers)==0):
#                 refers = [get_spepc(hps, ref_wav_path).to(dtype).to(device)]
#         else:
#             refer = get_spepc(hps, ref_wav_path).to(device).to(dtype)

#     t1 = ttime()
#     # os.environ['version'] = version
#     prompt_language = dict_language[prompt_language.lower()]
#     text_language = dict_language[text_language.lower()]
#     phones1, bert1, norm_text1 = get_phones_and_bert(prompt_text, prompt_language, version)
#     texts = text.split("\n")
#     audio_bytes = BytesIO()

#     for text in texts:
#         # 简单防止纯符号引发参考音频泄露
#         if only_punc(text):
#             continue

#         audio_opt = []
#         if (text[-1] not in splits): text += "。" if text_language != "en" else "."
#         phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)
#         bert = torch.cat([bert1, bert2], 1)

#         all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
#         bert = bert.to(device).unsqueeze(0)
#         all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
#         t2 = ttime()
#         with torch.no_grad():
#             pred_semantic, idx = t2s_model.model.infer_panel(
#                 all_phoneme_ids,
#                 all_phoneme_len,
#                 prompt,
#                 bert,
#                 # prompt_phone_len=ph_offset,
#                 top_k = top_k,
#                 top_p = top_p,
#                 temperature = temperature,
#                 early_stop_num=hz * max_sec)
#             pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
#         t3 = ttime()

#         if version != "v3":
#             audio = \
#                 vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0),
#                                 refers,speed=speed).detach().cpu().numpy()[
#                     0, 0]  ###试试重建不带上prompt部分
#         else:
#             phoneme_ids0=torch.LongTensor(phones1).to(device).unsqueeze(0)
#             phoneme_ids1=torch.LongTensor(phones2).to(device).unsqueeze(0)
#             # print(11111111, phoneme_ids0, phoneme_ids1)
#             fea_ref,ge = vq_model.decode_encp(prompt.unsqueeze(0), phoneme_ids0, refer)
#             ref_audio, sr = torchaudio.load(ref_wav_path)
#             ref_audio=ref_audio.to(device).float()
#             if (ref_audio.shape[0] == 2):
#                 ref_audio = ref_audio.mean(0).unsqueeze(0)
#             if sr!=24000:
#                 ref_audio=resample(ref_audio,sr)
#             # print("ref_audio",ref_audio.abs().mean())
#             mel2 = mel_fn(ref_audio)
#             mel2 = norm_spec(mel2)
#             T_min = min(mel2.shape[2], fea_ref.shape[2])
#             mel2 = mel2[:, :, :T_min]
#             fea_ref = fea_ref[:, :, :T_min]
#             if (T_min > 468):
#                 mel2 = mel2[:, :, -468:]
#                 fea_ref = fea_ref[:, :, -468:]
#                 T_min = 468
#             chunk_len = 934 - T_min
#             # print("fea_ref",fea_ref,fea_ref.shape)
#             # print("mel2",mel2)
#             mel2=mel2.to(dtype)
#             fea_todo, ge = vq_model.decode_encp(pred_semantic, phoneme_ids1, refer, ge,speed)
#             # print("fea_todo",fea_todo)
#             # print("ge",ge.abs().mean())
#             cfm_resss = []
#             idx = 0
#             while (1):
#                 fea_todo_chunk = fea_todo[:, :, idx:idx + chunk_len]
#                 if (fea_todo_chunk.shape[-1] == 0): break
#                 idx += chunk_len
#                 fea = torch.cat([fea_ref, fea_todo_chunk], 2).transpose(2, 1)
#                 # set_seed(123)
#                 cfm_res = vq_model.cfm.inference(fea, torch.LongTensor([fea.size(1)]).to(fea.device), mel2, sample_steps, inference_cfg_rate=0)
#                 cfm_res = cfm_res[:, :, mel2.shape[2]:]
#                 mel2 = cfm_res[:, :, -T_min:]
#                 # print("fea", fea)
#                 # print("mel2in", mel2)
#                 fea_ref = fea_todo_chunk[:, :, -T_min:]
#                 cfm_resss.append(cfm_res)
#             cmf_res = torch.cat(cfm_resss, 2)
#             cmf_res = denorm_spec(cmf_res)
#             if bigvgan_model==None:init_bigvgan()
#             with torch.inference_mode():
#                 wav_gen = bigvgan_model(cmf_res)
#                 audio=wav_gen[0][0].cpu().detach().numpy()

#         max_audio=np.abs(audio).max()
#         if max_audio>1:
#             audio/=max_audio
#         audio_opt.append(audio)
#         audio_opt.append(zero_wav)
#         audio_opt = np.concatenate(audio_opt, 0)
#         t4 = ttime()

#         sr = hps.data.sampling_rate if version != "v3" else 24000
#         if if_sr and sr == 24000:
#             audio_opt = torch.from_numpy(audio_opt).float().to(device)
#             audio_opt,sr=audio_sr(audio_opt.unsqueeze(0),sr)
#             max_audio=np.abs(audio_opt).max()
#             if max_audio > 1: audio_opt /= max_audio
#             sr = 48000

#         if is_int32:
#             audio_bytes = pack_audio(audio_bytes,(audio_opt * 2147483647).astype(np.int32),sr)
#         else:
#             audio_bytes = pack_audio(audio_bytes,(audio_opt * 32768).astype(np.int16),sr)
#     # logger.info("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
#         if stream_mode == "normal":
#             audio_bytes, audio_chunk = read_clean_buffer(audio_bytes)
#             yield audio_chunk
    
#     if not stream_mode == "normal": 
#         if media_type == "wav":
#             sr = 48000 if if_sr else 24000
#             sr = hps.data.sampling_rate if version != "v3" else sr
#             audio_bytes = pack_wav(audio_bytes,sr)
#         yield audio_bytes.getvalue()


# # ref_wav_path = 'ref.wav'
# # ref_text = 'ref_text'
# # text = 'text'

# # def get_tts(ref_wav_path, ref_text, text):
# #     for msg in get_tts_wav(
# #         ref_wav_path, 
# #         ref_text, 
# #         'Chinese',
# #         text, 
# #         'Chinese',
# #         '不切',
# #         15,
# #         1,
# #         1, 
# #         False,
# #         1,
# #         False, 
# #         None,
# #         32,
# #         False,
# #         0.3):
# #         print(msg)

# def get_tts(ref_wav_path, ref_text, text):
#     audio_data = get_tts_wav2(
#         ref_wav_path, 
#         ref_text, 
#         'Chinese',
#         text, 
#         'Chinese',
#         15,
#         1,
#         1, 
#         1,
#         None,
#         32,
#         False,
#         spk='default')
#     ic(audio_data)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--ref_wav_path', type=str, required=True)
#     parser.add_argument('--ref_text', type=str, required=True)
#     parser.add_argument('--text', type=str, required=True)
#     args = parser.parse_args()
#     ref_wav_path = args.ref_wav_path
#     ref_text = args.ref_text
#     text = args.text
#     get_tts(ref_wav_path, ref_text, text)
#     print('done')