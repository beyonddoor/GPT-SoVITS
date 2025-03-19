#! /bin/bash

start() {
    nohup celery -A worker.celery_app worker --loglevel=info &> /data/sovits/logs/celery.log &
    nohup uvicorn main:app --port 8001 --reload --loglevel=debug &> /data/sovits/logs/uvicorn.log &
    nohup redis-server &
}

stop() {
    pkill -9 uvicorn
    pkill -9 celery 
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        stop
        start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac