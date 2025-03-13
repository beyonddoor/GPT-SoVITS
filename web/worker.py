import time
from celeryconfig import celery_app
from subprocess import Popen
import subprocess
import os
import logging
from icecream import ic

PYTHON_CMD = 'python3'

def make_log_dir(user_id):
    log_dir = f'run/log_{user_id}/'
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

@celery_app.task(bind=True)
def pipeline_train(self, user_id: str):
    try:
        start_time = time.time()

        log_dir = make_log_dir(user_id)
        task_id = self.request.id
        logging.info("Start training for user %s, %s", user_id, task_id)

        process = subprocess.Popen([PYTHON_CMD, 'pipeline_train.py', '-id', user_id],
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

        return f"Task Completed, run time {run_time}"
    except Exception as e:
        logging.error("Error: %s", e)
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise


@celery_app.task(bind=True)
def pipeline_infer(self, user_id, text):

    start_time = time.time()
    log_dir = make_log_dir(user_id)

    process = subprocess.Popen([PYTHON_CMD, 'GPT_SoVITS/inference_cli.py', '-i', f'data/{user_id}/input_audio', '-o', f'output/{user_id}/']
                               , stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd='..', shell=False)
    stdout, stderr = process.communicate()

    date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    with open(f'{log_dir}/info_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    with open(f'{log_dir}/error_{date}.log', 'w') as f:
        f.write(stdout.decode())
    
    run_time = time.time() - start_time
    return f"Task Completed, exit code {process.returncode}, run time {run_time}"

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("Worker started")