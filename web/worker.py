# from ..hook_proc import hook_proc

import time
from celeryconfig import celery_app
from subprocess import Popen
import subprocess
import os
import logging
from icecream import ic
import webconfig

PYTHON_CMD = 'python3'

def make_log_dir(voice_id, prefix):
    log_dir = f'{webconfig.log_dir}/{prefix}/{voice_id}/'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@celery_app.task(bind=True)
def pipeline_train(self, voice_id: str, unique_id: str):
    start_time = time.time()
    try:
        log_dir = make_log_dir(voice_id, 'train')
        task_id = self.request.id
        logging.info("Start training for user %s, %s, unique_id:%s", voice_id, task_id, unique_id)

        cmds = [PYTHON_CMD, 'pipeline_train.py', '-id', voice_id]

        # 假的训练，方便在没有训练能力的环境测试
        if webconfig.dummy_train:
            cmds.append('--dummy')

        process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..', shell=False)
        
        stdout, stderr = process.communicate()
        with open(f'{log_dir}/info_{unique_id}.log', 'w') as f:
            f.write(stdout.decode())

        with open(f'{log_dir}/error_{unique_id}.log', 'w') as f:
            f.write(stderr.decode())

        run_time = time.time() - start_time
        if process.returncode != 0:
            raise Exception('tranining failed, return code: %d, task_id=%s' % (process.returncode, task_id))

        return {
            'error': 0,
            'result': '',  #TODO 增加模型的位置
            'desc': f"Task Completed, time={run_time}, task_id={task_id}, unique_id={unique_id}"
        }
    except Exception as e:
        run_time = time.time() - start_time
        logging.error("Error: %s", e)
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise
        return {
            'error':1,
            'desc': f"Task Failed, time={run_time}, task_id={task_id}, unique_id={unique_id} {e}"
        }
    
def get_model(voice_id):
    '''返回模型的path'''
    if webconfig.dummy_train:
        return 'GPT_weights_v2/GPT-SoVITS-e10.ckpt', 'SoVITS_weights_v2/GPT-SoVITS_e4_s60.pth'
    
    # TODO 实现这个，暂时找第一个模型
    model_dir = os.path.join(webconfig.model_dir, voice_id)
    files = os.listdir(model_dir)

    gpt_path = [f for f in files if f.endswith('.ckpt')][0]
    sovits_path = [f for f in files if f.endswith('.pth')][0]
    return os.path.join(model_dir, gpt_path), os.path.join(model_dir, sovits_path)

@celery_app.task(bind=True)
def pipeline_infer(self, voice_id, text, unique_id):
    '''
    # python GPT_SoVITS/inference_cli.py --gpt_model GPT_weights_v2/GPT-SoVITS-e10.ckpt 
    # --sovits_model SoVITS_weights_v2/GPT-SoVITS_e4_s60.pth 
    # --ref_audio data/2/infer/你好晚安.m4a --ref_text data/2/infer/你好晚安.txt 
    # --ref_language 中文 --target_text data/2/infer/infer.txt --target_language 中文 --output_path output2
    '''
    print('unique_id: ', unique_id)
    
    try:
        start_time = time.time()
        log_dir = make_log_dir(voice_id, 'infer')
        
        final_path = f'{webconfig.output_dir}/{voice_id}/{unique_id}'
        os.makedirs(f'{final_path}/', exist_ok=True)    
        
        temp_text_path = f'{final_path}.txt'
        with open(temp_text_path, 'w', encoding='utf8') as f:
            f.write(text)

        data_dir = os.path.join(webconfig.input_dir, voice_id)

        gpt_model, sovits_model = get_model(voice_id)

        cmds = [PYTHON_CMD, 'GPT_SoVITS/inference_cli.py', 
                                    '--gpt_model', gpt_model,
                                    '--sovits_model', sovits_model,
                                    '--ref_audio', os.path.join(data_dir, 'audio.wav'), 
                                    '--ref_text', os.path.join(data_dir, 'audio.txt'), 
                                    '--ref_language', '中文',
                                    '--target_text', temp_text_path,
                                    '--target_language', '中文',
                                    '--output_path', f'{final_path}/',
                                    ]
        print(cmds)
        process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..', shell=False)
        stdout, stderr = process.communicate()

        date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        with open(f'{log_dir}/info_{unique_id}.log', 'w') as f:
            f.write(stdout.decode())
        
        with open(f'{log_dir}/error_{unique_id}.log', 'w') as f:
            f.write(stderr.decode())
        
        run_time = time.time() - start_time
        if process.returncode == 0:
            if os.path.exists(f'{final_path}/output.wav'):
                return {
                    'error': 0,
                    'result': f'{voice_id}/{unique_id}/output.wav',
                    'desc': f"Task Completed, time={run_time}, unique_id={unique_id}"
                }
            else:
                return {
                    'error': 1,
                    'desc': f"Task Completed, time={run_time}, unique_id={unique_id}, {final_path}/output.wav not found"
                }
        return {
            'error': 1,
            'desc': f"Task Failed, time={run_time} retcode={process.returncode} unique_id={unique_id}"
        }
    except Exception as e:
        print(e)
        run_time = time.time() - start_time
        return {
            'error': 1,
            'desc': f"Task Failed, time={run_time}, unique_id={unique_id}, {e}"
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Worker started")