## 安装
pip install fastapi uvicorn celery redis

## 启动
nohup celery -A worker.celery_app worker --loglevel=info &
nohup uvicorn main:app --reload &
nohup redis-server &

## 测试
curl -X 'POST' 'http://127.0.0.1:8000/train/' -H 'Content-Type: application/json' -d '{"user_id": "1"}'
curl -X 'GET' 'http://127.0.0.1:8000/train/some-task-id'



python GPT_SoVITS/inference_cli.py --gpt_model GPT_weights_v2/GPT-SoVITS-e10.ckpt --sovits_model SoVITS_weights_v2/GPT-SoVITS_e4_s60.pth --ref_audio data/2/infer/你好晚安.m4a --ref_text data/2/infer/你好晚安.txt --ref_language 中文 --target_text data/2/infer/infer.txt --target_language 中文 --output_path output


