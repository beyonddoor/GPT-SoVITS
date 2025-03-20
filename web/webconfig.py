
# infer输出目录
# {output_dir}/{voice_id}/{unique_id}.txt
# {output_dir}/{voice_id}/{unique_id}/
output_dir = '/data/sovits/output'  # uuid.wav

# 训练使用目录，包含audio和参考文本，{input_dir}/{voice_id}/audio.wav 和 audio.txt
input_dir = '/data/sovits/input'    # voice_id.wav

# GPT和SOVITS存放到{model_dir}/{voice_id}/目录中
model_dir = '/data/sovits/model'

# 日志目录
log_dir = '/data/sovits/logs'

download_safe_dir = output_dir

# 跳过训练
dummy_train = True