
## 设计

http://127.0.0.1:8000/train/

### 开始训练
POST http://127.0.0.1:8000/train
curl -X POST http://127.0.0.1:8001/train/  -F 'file=@audio.wav' -F 'json_data={"voice_id":"1", "text":"对应的文本"}' -H "Content-Type: multipart/form-data"
curl -X POST http://127.0.0.1:8070/ai/test/sovits/train/  -F 'voiceFile=@audio.wav' -H "Content-Type: multipart/form-data"

输入
```json
{
    "voice_id": "",
    "text": ""
}
同时包含audio文件
```

TaskStatus: TaskStarted | TaskRunning | TaskFinish | TaskFail

输出
```json
{
    "task_id": "",
    "status": "",  // TaskStatus
    "desc": "",  // 调试
}
```

### 查询训练结果
GET http://127.0.0.1:8000/train/$task_id
curl http://127.0.0.1:8070/ai/test/sovits/train/$task_id

输出
```json
{
    "task_id": "",
    "status": "",
    "result" : "",  // 正常输出
    "desc": "",  // 调试
}
```


### 开始推理
POST http://127.0.0.1:8000/infer
curl -X POST http://127.0.0.1:8001/infer/  -d '{"voice_id":"1", "text":"今年春天，孩子们在房前空地上，斩草挖土，开辟出来了一个一丈见方的小花园。周围用竹竿扎了一个篱笆，移来了一棵玉兰花树"}' \
-H Content-Type: application/json"

curl -X POST http://127.0.0.1:8070/ai/test/sovits/infer/  -d '{"voiceId":"1", "text":"今年春天，孩子们在房前空地上，斩草挖土，开辟出来了一个一丈见方的小花园。周围用竹竿扎了一个篱笆，移来了一棵玉兰花树"}' \
-H Content-Type: application/json"


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
    "result" : "",  // 正常输出
    "desc": "",  // 调试
}
```

### 查询推理结果
GET http://127.0.0.1:8000/infer/${task_id}

输出
```json
{
    "task_id": "",
    "status": "",
    "result" : "",  // 正常输出
    "desc": "",  // 调试
}
```