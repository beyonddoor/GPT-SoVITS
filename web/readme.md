pip install fastapi uvicorn celery redis

celery -A worker.celery_app worker --loglevel=info
uvicorn main:app --reload
redis-server

curl -X 'POST' 'http://127.0.0.1:8000/start-task/' -H 'Content-Type: application/json'
curl -X 'GET' 'http://127.0.0.1:8000/task-status/some-task-id'

