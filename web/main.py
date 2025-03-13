from fastapi import FastAPI
from celery.result import AsyncResult
from worker import pipeline_train, pipeline_infer
from celeryconfig import celery_app
from pydantic import BaseModel

app = FastAPI()

class TrainRequest(BaseModel):
    user_id: str

class InferRequest(BaseModel):
    user_id: str
    text: str

@app.post("/train/")
def train(data:TrainRequest):
    task = pipeline_train.delay(data.user_id)
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
def infer(data: InferRequest):
    task = pipeline_infer.delay(data.user_id, data.text)
    return {"task_id": task.id, "status": "Task started"}

@app.get("/infer/{task_id}")
def get_infer_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else "Processing..."
    }

@app.post("/infer2/{task_id}")
def infer2(task_id: str):
    pass
    # python GPT_SoVITS/inference_cli.py --gpt_model GPT_weights_v2/GPT-SoVITS-e10.ckpt --sovits_model SoVITS_weights_v2/GPT-SoVITS_e4_s60.pth --ref_audio data/2/infer/你好晚安.m4a --ref_text data/2/infer/你好晚安.txt --ref_language 中文 --target_text data/2/infer/infer.txt --target_language 中文 --output_path output
