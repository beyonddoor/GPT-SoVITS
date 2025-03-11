import time
from celeryconfig import celery_app

@celery_app.task(bind=True)
def long_running_task(self):
    for i in range(10):  # Simulate long task (10 minutes)
        time.sleep(60)   # Sleep 1 minute per iteration
    return "Task Completed"

