import time
from celeryconfig import celery_app
from subprocess import Popen
import subprocess
import os
import logging
from icecream import ic
import webconfig

PYTHON_CMD = 'python3'

def make_log_dir(voice_id):
    log_dir = f'{webconfig.log_dir}/{voice_id}/'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@celery_app.task(bind=True)
def pipeline_train(self, voice_id: str):
    try:
        start_time = time.time()

        log_dir = make_log_dir(voice_id)
        task_id = self.request.id
        logging.info("Start training for user %s, %s", voice_id, task_id)

        process = subprocess.Popen([PYTHON_CMD, 'pipeline_train.py', '-id', voice_id],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..', shell=False)
        stdout, stderr = process.communicate()

        date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        with open(f'{log_dir}/info_{date}_{self.request.id}.log', 'w') as f:
            f.write(stdout.decode())

        with open(f'{log_dir}/error_{date}.log', 'w') as f:
            f.write(stderr.decode())

        run_time = time.time() - start_time
        if process.returncode != 0:
            raise Exception('tranining failed, return code: %d, task_id=%s' % (process.returncode, task_id))

        return {
            'error': 0,
            'result': f"Task Completed, run time {run_time}"
        }
    except Exception as e:
        logging.error("Error: %s", e)
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise
        return {
            'error':1,
            'result': str(e)
        }

@celery_app.task(bind=True)
def pipeline_infer(self, voice_id, text):

    start_time = time.time()
    log_dir = make_log_dir(voice_id)
    
    os.makedirs(f'{webconfig.output_dir}/{voice_id}/', exist_ok=True)

    process = subprocess.Popen([PYTHON_CMD, 'GPT_SoVITS/inference_cli.py', '-i', f'{webconfig.input_dir}/{voice_id}', '-o', f'{webconfig.output_dir}/{voice_id}/']
                               , stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..', shell=False)
    stdout, stderr = process.communicate()

    date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    with open(f'{log_dir}/info_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    with open(f'{log_dir}/error_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    run_time = time.time() - start_time
    if process.returncode == 0:
        return {
            'error': 0,
            'result': f"Task Completed, run time {run_time}"
        }
    return {
        'error': 1,
        'result': f"Task Failed, run time {run_time} {process.returncode}"
    }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Worker started")