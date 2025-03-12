import time
from celeryconfig import celery_app
from subprocess import Popen
import subprocess
import os

def make_log_dir(user_id):
    log_dir = f'run/log_{user_id}/'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@celery_app.task(bind=True)
def pipeline_train(self, user_id:str):

    start_time = time.time()
    log_dir = make_log_dir(user_id)

    process = subprocess.Popen(['python', '../pipeline_train.py', '-id', user_id]
                               , stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..')
    stdout, stderr = process.communicate()

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(f'{log_dir}/info_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    with open(f'{log_dir}/error_{date}.log', 'w') as f:
        f.write(stdout.decode())

    run_time = time.time() - start_time
    return f"Task Completed, exit code {process.returncode}, run time {run_time}"


@celery_app.task(bind=True)
def pipeline_infer(self, user_id, text):

    start_time = time.time()
    log_dir = make_log_dir(user_id)

    process = subprocess.Popen(['python', '../GPT_SoVITS/inference_cli.py', '-i', f'data/{user_id}/input_audio', '-o', f'output/{user_id}/']
                               , stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..')
    stdout, stderr = process.communicate()

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(f'{log_dir}/info_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    with open(f'{log_dir}/error_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    run_time = time.time() - start_time
    return f"Task Completed, exit code {process.returncode}, run time {run_time}"