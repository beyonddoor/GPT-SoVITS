## 安装
pip install fastapi uvicorn celery redis

## 启动
nohup celery -A worker.celery_app worker --loglevel=info &
nohup uvicorn main:app --reload &
nohup redis-server &

## 测试
curl -X 'POST' 'http://127.0.0.1:8000/train/new' -H 'Content-Type: application/json' -d '{"voice_id": "1"}'
curl -X 'GET' 'http://127.0.0.1:8000/train/some-task-id'

代理

python GPT_SoVITS/inference_cli.py --gpt_model GPT_weights_v2/GPT-SoVITS-e10.ckpt --sovits_model SoVITS_weights_v2/GPT-SoVITS_e4_s60.pth --ref_audio data/2/infer/你好晚安.m4a --ref_text data/2/infer/你好晚安.txt --ref_language 中文 --target_text data/2/infer/infer.txt --target_language 中文 --output_path output

## todo
验证gradio的端口和工作机制
    后续看不到启动gradio的功能了
~~验证autodl的内网架构。lsof看不到非本地端点连接22端口。不知道是怎么做到的~~

sovits训练输出到
logs/GPT-SoVITS/logs_s2_v2

gpt训练输出到
logs/GPT-SoVITS/logs_s1_v2/

这些缓存如果不clear，会跳过训练

对比环境变量
对比配置文件yaml和json

pandas显示不连续的时序
增加指定的cmd的统计进行比较，而不是全量
增加unkown协议的打印

## 待解决问题
1. temp路径被remove的问题
1. cli推理下来语音模糊的问题
2. pipeline_train处理下来vocal文件没有生成的问题。id=1

~~似乎和用户dir分离有关~~
这个会偶然发生，原来的webui也可能有这个问题

clean_empty_cache
==> 1.m4a.reformatted.wav->Traceback (most recent call last):
  File "/root/GPT-SoVITS/tools/uvr5/webui.py", line 294, in uvr
    pre_fun._path_audio_(
  File "/root/GPT-SoVITS/tools/uvr5/vr.py", line 166, in _path_audio_
    wav_vocals = spec_utils.cmb_spectrogram_to_wave(
  File "/root/GPT-SoVITS/tools/uvr5/lib/lib_v5/spec_utils.py", line 400, in cmb_spectrogram_to_wave
    wave = librosa.resample(
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/librosa/util/decorators.py", line 88, in inner_f
    return f(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/librosa/core/audio.py", line 606, in resample
    util.valid_audio(y, mono=False)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/librosa/util/decorators.py", line 88, in inner_f
    return f(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/librosa/util/utils.py", line 294, in valid_audio
    raise ParameterError("Audio buffer is not finite everywhere")
librosa.util.exceptions.ParameterError: Audio buffer is not finite everywhere

3. 第二阶段训练时报错
ckpt_path为none所致

/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py:293: The number of training batches (15) is smaller than the logging interval Trainer(log_every_n_steps=50). Set a lower value for log_every_n_steps if you want to see logs for the training epoch.
Epoch 0:   0%|                                                         | 0/15 [00:00<?, ?it/s]Traceback (most recent call last):
  File "/root/GPT-SoVITS/GPT_SoVITS/s1_train.py", line 179, in <module>
    main(args)
  File "/root/GPT-SoVITS/GPT_SoVITS/s1_train.py", line 155, in main
    trainer.fit(model, data_module, ckpt_path=ckpt_path)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 544, in fit
    call._call_and_handle_interrupt(
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 44, in _call_and_handle_interrupt
    return trainer_fn(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 580, in _fit_impl
    self._run(model, ckpt_path=ckpt_path)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 989, in _run
    results = self._run_stage()
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py", line 1035, in _run_stage
    self.fit_loop.run()
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 202, in run
    self.advance()
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/fit_loop.py", line 359, in advance
    self.epoch_loop.run(self._data_fetcher)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 136, in run
    self.advance(data_fetcher)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/training_epoch_loop.py", line 242, in advance
    batch_output = self.manual_optimization.run(kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/manual.py", line 92, in run
    self.advance(kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/loops/optimization/manual.py", line 112, in advance
    training_step_output = call._call_strategy_hook(trainer, "training_step", *kwargs.values())
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/trainer/call.py", line 309, in _call_strategy_hook
    output = fn(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/pytorch_lightning/strategies/strategy.py", line 382, in training_step
    return self.lightning_module.training_step(*args, **kwargs)
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/models/t2s_lightning_module.py", line 39, in training_step
    loss, acc = forward(
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/models/t2s_model.py", line 463, in forward_old
    xy_dec, _ = self.h(
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/modules/transformer.py", line 170, in forward
    output = mod(
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/modules/transformer.py", line 311, in forward
    x + self._sa_block(x, src_mask, src_key_padding_mask, cache=cache),
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/modules/transformer.py", line 332, in _sa_block
    x = self.self_attn(
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/root/miniconda3/envs/GPTSoVits/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1527, in _call_impl
    return forward_call(*args, **kwargs)
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/modules/activation.py", line 404, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
  File "/root/GPT-SoVITS/GPT_SoVITS/AR/modules/patched_mha_with_cache.py", line 452, in multi_head_attention_forward_patched
    attn_output = scaled_dot_product_attention(
RuntimeError: Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: float and  query.dtype: c10::BFloat16 instead.
Epoch 0:   0%|          | 0/15 [00:00<?, ?it/s]
('GPT Training Finished', {'__type__': 'update', 'visible': True}, {'__type__': 'update', 'visible': False})



