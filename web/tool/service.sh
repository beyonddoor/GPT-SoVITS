#! /bin/bash

script_dir="$(cd "$(dirname "$0")" && pwd)"

start() {
    cd "$script_dir/.." || exit 1

    nohup celery -A worker.celery_app worker --loglevel=info &>> /data/sovits/logs/celery.log &
    nohup uvicorn main:app --port 8001 --reload --log-level=debug &>> /data/sovits/logs/uvicorn.log &
    nohup redis-server &
}

stop() {
    # pkill -9 uvicorn
    local pid
    pid="$(netstat -ntlp | grep 127.0.0.1:8001 | rev | awk '{print $1}' | cut -d/ -f2 | rev)"
    kill -9 "$pid"
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