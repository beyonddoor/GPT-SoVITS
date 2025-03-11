from celery import Celery

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",  # Redis as the broker
    backend="redis://localhost:6379/0"  # Store results in Redis
)

