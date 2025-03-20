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
from icecream import ic
import time

import logging.config
from logconfig import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

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
    # if result.state == 'PENDING' and not result.result:
    #     return False
    return True

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
    ic(json_data)
    try:
        logging.info('json_data %s', json_data)
        data = json.loads(json_data)
        unique_id = f"{data['voice_id']}_{int(time.time())}"

        upload_dir = os.path.join(webconfig.input_dir, data['voice_id'])
        os.makedirs(upload_dir, exist_ok=True)

        file_content = await file.read()
        with open(os.path.join(upload_dir, 'audio.wav'), 'wb') as f:
            f.write(file_content)

        with open(os.path.join(upload_dir, 'audio.txt'), 'w', encoding='utf8') as f:
            f.write(data['text'])

        task = pipeline_train.delay(data['voice_id'], unique_id)
        return {"task_id": task.id, "status": "TaskStarted", "result": ""}
        
    except json.JSONDecodeError as e:
        logging.error('error %s', e)
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logging.error('error %s', e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/train/{task_id}")
def get_train_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)

    ic(task_id, task_result)

    if not check_task_available(task_result):
        # 不存在
        return {
            "task_id": task_id,
            "status": TaskFail,
            "desc": "task unknown"
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
                "desc": result
            } 
            
    return {
        "task_id": task_id,
        "status": TaskRunning,
        "desc": f'status = {task_result.status}'
    }

@app.post("/infer/")
def infer(data: InferRequest):
    unique_id = f"{data.voice_id}_{int(time.time())}"
    task = pipeline_infer.delay(data.voice_id, data.text, unique_id)
    return {
        "task_id": task.id, 
        "status": "TaskStarted",
        "desc": f"unique_id = {unique_id}"
    }

@app.get("/infer/{task_id}")
def get_infer_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    ic(task_id, task_result.status, task_result.result, task_result.ready())
    
    if not check_task_available(task_result):
        # 不存在
        return {
            "task_id": task_id,
            "status": TaskFail,
            "desc": "task unknown"
        }
    
    if task_result.ready():
        if task_result.status == states.SUCCESS:
            # 任务完成
            result = task_result.result
            if result['error'] == 0:
                return {
                    "task_id": task_id,
                    "status": TaskFinish,
                    "result": result['result'],
                    "desc": result,
                }
                
            return {
                "task_id": task_id,
                "status": TaskFail,
                "desc": result
            } 
            
    return {
        "task_id": task_id,
        "status": TaskRunning,
        "desc": f'status = {task_result.status}'
    }

@app.get('/download/{filename:path}')
def download(filename: str):
    files_directory = webconfig.download_safe_dir
    safe_dir = os.path.abspath(files_directory)
    file_path = os.path.abspath(os.path.join(safe_dir, filename))
    logging.info('download %s %s', filename, file_path)
    ic(safe_dir, file_path, filename)
    
    if not file_path.startswith(safe_dir):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        filename=filename,
        media_type="application/octet-stream"
    )