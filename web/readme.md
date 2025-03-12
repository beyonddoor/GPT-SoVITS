## 安装
pip install fastapi uvicorn celery redis

## 启动
nohup celery -A worker.celery_app worker --loglevel=info &
nohup uvicorn main:app --reload &
nohup redis-server &

## 测试
curl -X 'POST' 'http://127.0.0.1:8000/train/' -H 'Content-Type: application/json' -d '{"user_id": "1"}'
curl -X 'GET' 'http://127.0.0.1:8000/task-status/some-task-id'

