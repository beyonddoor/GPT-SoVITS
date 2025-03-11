from fastapi import FastAPI
from celery.result import AsyncResult
from worker import long_running_task
from celeryconfig import celery_app

app = FastAPI()

@app.post("/start-task/")
def start_task():
    task = long_running_task.delay()  # Run task asynchronously
    return {"task_id": task.id, "status": "Task started"}

@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else "Processing..."
    }
