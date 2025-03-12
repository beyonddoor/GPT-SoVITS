from fastapi import FastAPI
from celery.result import AsyncResult
from worker import pipeline_train, pipeline_infer
from celeryconfig import celery_app

app = FastAPI()

@app.post("/train/")
def train(user_id:str):
    task = pipeline_train.delay(user_id)
    return {"task_id": task.id, "status": "Task started"}

@app.get("/train/{task_id}")
def get_train_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else "Processing..."
    }

@app.post("/infer/")
def infer(user_id:str, text:str):
    task = pipeline_infer.delay(user_id, text)
    return {"task_id": task.id, "status": "Task started"}

@app.get("/infer/{task_id}")
def get_train_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else "Processing..."
    }
