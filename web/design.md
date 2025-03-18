
## 设计

http://127.0.0.1:8000/train/

### 开始训练
POST http://127.0.0.1:8000/train

输入
```json
{
    "voice_id": ""
}
同时包含audio文件
```

TaskStatus: TaskStarted | TaskRunning | TaskFinish | TaskFail

输出
```json
{
    "task_id": "",
    "status": "",  // TaskStatus
    "result" : "",      // 如果是fail，则为error描述
}
```

### 查询训练结果
GET http://127.0.0.1:8000/train/$task_id

输出
```json
{
    "task_id": "",
    "status": "",
    "result" : "",
    // "model_path": ""    // 模型path根据voice_id来保存
}
```


### 开始推理
POST http://127.0.0.1:8000/infer

输入
```json
{
    "voice_id": "",
    "text": "",
}
```

输出
```json
{
    "task_id": "",
    "status": "",
    "result" : "",
}
```

### 查询推理结果
GET http://127.0.0.1:8000/infer/${task_id}

输出
```json
{
    "task_id": "",
    "status": "",
    "result" : "",
    "audio_path": ""
}
```