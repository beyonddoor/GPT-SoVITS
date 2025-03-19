from celery.result import AsyncResult
from celery import states

from worker import pipeline_train, pipeline_infer
from celeryconfig import celery_app
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os
import webconfig
import enum
from fastapi import File, UploadFile, Form
from typing import Optional
import json
import logging

import logging.config
from logconfig import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

# class TaskStatus(enum.Enum):
    # TaskStarted
    # TaskRunning
    # TaskFinish
    # TaskFail

TaskStarted = 'TaskStarted'
TaskRunning = 'TaskRunning'
TaskFinish = 'TaskFinish'
TaskFail = 'TaskFail'

app = FastAPI()

# class TrainRequest(BaseModel):
#     voice_id : str

class InferRequest(BaseModel):
    voice_id: str
    text: str

def check_task_available(result):
    '''task是否有效'''
    if result.state == 'PENDING' and not result.result:
        return False
    return True

# curl命令行调用示例:
# curl -X POST http://localhost:8000/train/ \
#   -F "file=@/path/to/audio.wav" \
#   -F "json_data={\"voice_id\":\"test_voice_001\"}" \
#   -H "Content-Type: multipart/form-data"

@app.post("/train/")
async def start_train(
    file: UploadFile = File(...),
    json_data: str = Form(...)
):
    """
    处理文件上传和JSON数据的POST请求
    file: 上传的文件
    json_data: JSON格式的字符串数据
    """
    try:
        logging.info('json_data %s', json_data)
        data = json.loads(json_data)
        file_content = await file.read()
        filename = 'audio.wav'
        content_type = file.content_type
        upload_dir = os.path.join(webconfig.input_dir, data['voice_id'])
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
            
        task = pipeline_train.delay(data['voice_id'])
        return {"task_id": task.id, "status": "TaskStarted", "result": ""}
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{task_id}")
def get_train_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    if not check_task_available(task_result):
        # 不存在
        return {
            "task_id": task_id,
            "status": TaskFail,
            "result": "task unknown"
        }
    
    if task_result.ready():
        if task_result.status == states.SUCCESS:
            
            # 任务完成
            result = task_result.result
            if result['error'] == 0:
                return {
                    "task_id": task_id,
                    "status": TaskFinish,
                    "result": result['result']
                }
                
            return {
                "task_id": task_id,
                "status": TaskFail,
                "result": result
            } 
            
    return {
        "task_id": task_id,
        "status": TaskRunning,
        "result": f'status = {task_result.status}'
    }

@app.post("/infer/")
def infer(data: InferRequest):
    task = pipeline_infer.delay(data.voice_id, data.text)
    return {
        "task_id": task.id, 
        "status": "TaskStarted",
        "result": ""
    }

@app.get("/infer/{task_id}")
def get_infer_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    # TODO 完善
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else "Processing...",
        "audio_path": ""
    }

# @app.post("/infer2/{task_id}")
# def infer2(task_id: str):
#     pass
#     # python GPT_SoVITS/inference_cli.py --gpt_model GPT_weights_v2/GPT-SoVITS-e10.ckpt --sovits_model SoVITS_weights_v2/GPT-SoVITS_e4_s60.pth --ref_audio data/2/infer/你好晚安.m4a --ref_text data/2/infer/你好晚安.txt --ref_language 中文 --target_text data/2/infer/infer.txt --target_language 中文 --output_path output


@app.get('/download/{filename}')
def download(filename: str):
    files_directory = webconfig.download_safe_dir
    safe_dir = os.path.abspath(files_directory)
    file_path = os.path.abspath(os.path.join(safe_dir, filename))
    
    if not file_path.startswith(safe_dir):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/octet-stream"
    )